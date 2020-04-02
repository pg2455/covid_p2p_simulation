import datetime
import random

import numpy as np

from config import BASELINE_P_ASYMPTOMATIC, P_COLD, P_FLU, \
    P_TEST, P_FALSE_NEGATIVE, SYMPTOM_DAYS, INFECTIOUSNESS_CURVE, AVERAGE_INCUBATION_DAYS, SCALE_INCUBATION_DAYS
from event import Event
from utils import _draw_random_discreet_gaussian


class Infection(object):

    def __init__(self, agent):
        self.human = agent

        # probability of being asymptomatic is basically 50%, but a bit less if you're older
        # and a bit more if you're younger
        self.asymptomatic = np.random.rand() > (BASELINE_P_ASYMPTOMATIC - (self.age - 50) * 0.5) / 100
        self.incubation_days = _draw_random_discreet_gaussian(AVERAGE_INCUBATION_DAYS, SCALE_INCUBATION_DAYS)

        self.has_cold = np.random.rand() < P_COLD * self.age_modifier
        self.has_flu = np.random.rand() < P_FLU * self.age_modifier

        # Indicates whether this person will show severe signs of illness.
        self.really_sick = self.is_sick and random.random() >= 0.9
        self.never_recovers = random.random() >= 0.99

        self.update_health()

    @property
    def test_results(self):
        if self.symptoms == None:
            return None
        else:
            tested = np.random.rand() > P_TEST
            if tested:
                if self.is_sick:
                    return 'positive'
                else:
                    if np.random.rand() > P_FALSE_NEGATIVE:
                        return 'negative'
                    else:
                        return 'positive'
            else:
                return None

    @property
    def reported_symptoms(self):
        if self.symptoms is None or self.test_results is None or not self.human.has_app:
            return None
        else:
            if np.random.rand() < self.carefullness:
                return self.symptoms
            else:
                return None

    @property
    def symptoms(self):
        # probability of being asymptomatic is basically 50%, but a bit less if you're older
        # and a bit more if you're younger
        symptoms = None
        if self.asymptomatic or self.infection_timestamp is None:
            pass
        else:
            time_since_sick = self.env.timestamp - self.infection_timestamp
            symptom_start = datetime.timedelta(abs(np.random.normal(SYMPTOM_DAYS, 2.5)))
            #  print (time_since_sick)
            #  print (symptom_start)
            if time_since_sick >= symptom_start:
                symptoms = ['mild']
                if self.really_sick:
                    symptoms.append('severe')
                if np.random.rand() < 0.9:
                    symptoms.append('fever')
                if np.random.rand() < 0.85:
                    symptoms.append('cough')
                if np.random.rand() < 0.8:
                    symptoms.append('fatigue')
                if np.random.rand() < 0.7:
                    symptoms.append('trouble_breathing')
                if np.random.rand() < 0.1:
                    symptoms.append('runny_nose')
                if np.random.rand() < 0.4:
                    symptoms.append('loss_of_taste')
                if np.random.rand() < 0.4:
                    symptoms.append('gastro')
        if self.has_cold:
            if symptoms is None:
                symptoms = ['mild', 'runny_nose']
            if np.random.rand() < 0.2:
                symptoms.append('fever')
            if np.random.rand() < 0.6:
                symptoms.append('cough')
        if self.has_flu:
            if symptoms is None:
                symptoms = ['mild']
            if np.random.rand() < 0.2:
                symptoms.append('severe')
            if np.random.rand() < 0.8:
                symptoms.append('fever')
            if np.random.rand() < 0.4:
                symptoms.append('cough')
            if np.random.rand() < 0.8:
                symptoms.append('fatigue')
            if np.random.rand() < 0.8:
                symptoms.append('aches')
            if np.random.rand() < 0.5:
                symptoms.append('gastro')
        return symptoms

    @property
    def infectiousness(self):
        if self.is_sick:
            days_sick = (self.env.timestamp - self.infection_timestamp).days
            if days_sick > len(INFECTIOUSNESS_CURVE):
                return 0
            else:
                return INFECTIOUSNESS_CURVE[days_sick - 1]
        else:
            return 0

    @property
    def is_contagious(self):
        return self.infectiousness

    def run(self):
        while True:
            # FIXME: let update the health dynamically
            self.update_health()
            if self.is_sick and self.env.timestamp - self.infection_timestamp > datetime.timedelta(
                    days=self.incubation_days):
                # Todo ensure it only happen once
                result = random.random() > 0.8
                Event.log_test(self.human, time=self.env.timestamp, result=result)
                # Fixme: After a user get tested positive, assume no more activity
                break
            yield self.env.timeout(30)

    def update_health(self):
        self.human.health = {
            'is_infected': self.is_sick,
            'infection_timestamp': self.infection_timestamp,
            'infectiousness': self.infectiousness,
            'reported_symptoms': self.reported_symptoms,
            'symptoms': self.symptoms,
            'test_results': self.test_results,
        }

    # Extend the Shared property of human
    @property
    def infection_timestamp(self):
        return self.human.infection_timestamp

    @property
    def age(self):
        return self.human.age

    @property
    def env(self):
        return self.human.env

    @property
    def carefullness(self):
        return self.human.carefullness

    @property
    def is_sick(self):
        return self.human.is_sick

    @property
    def age_modifier(self):
        return self.human.age_modifier
