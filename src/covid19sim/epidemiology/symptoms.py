import math
from collections import namedtuple, OrderedDict


"""
---------------------------------------
-------------SYMPTOMS -----------------
---------------------------------------
"""

SymptomProbability = namedtuple('SymptomProbability', ['name', 'id', 'probabilities'])
SymptomProbability.__doc__ = '''A symptom probabilities collection given contexts

Attributes
    ----------
    name : str
        name of the symptom
    id : positive int
        id of the symptom
        This attribute should never change once set. It is used to define the
        position of the symptom in a multi-hot encoding
    probabilities : dict
        probabilities of the symptom per context
        A probability of `-1` is assigned when it is heavily variable given
        multiple factors and is handled entirely in the code
        A probability of `None` is assigned when the symptom can be skipped
        entirely in the context. The context can also be removed from the dict
'''

SYMPTOMS_CONTEXTS = {'covid': {0: 'covid_incubation', 1: 'covid_onset', 2: 'covid_plateau',
                               3: 'covid_post_plateau_1', 4: 'covid_post_plateau_2'},
                     'allergy': {0: 'allergy'},
                     'cold': {0: 'cold', 1: 'cold_last_day'},
                     'flu': {0: 'flu_first_day', 1: 'flu', 2: 'flu_last_day'}}

SYMPTOMS = OrderedDict([
    # Sickness severity
    # A lot of symptoms are dependent on the sickness severity so severity
    # level needs to be first
    (
        'mild',
        SymptomProbability('mild', 1, {'covid_incubation': 0.0,
                                       'covid_onset': -1,
                                       'covid_plateau': -1,
                                       'covid_post_plateau_1': -1,
                                       'covid_post_plateau_2': -1,
                                       'cold': -1,
                                       'cold_last_day': 1.0,
                                       'flu_first_day': 1.0,
                                       'flu': -1,
                                       'flu_last_day': 1.0})
    ),
    (
        'moderate',
        SymptomProbability('moderate', 0, {'covid_incubation': 0.0,
                                           'covid_onset': -1,
                                           'covid_plateau': -1,
                                           'covid_post_plateau_1': -1,
                                           'covid_post_plateau_2': -1,
                                           'cold': -1,
                                           'cold_last_day': 0.0,
                                           'flu_first_day': 0.0,
                                           'flu': -1,
                                           'flu_last_day': 0.0})
    ),
    (
        'severe',
        SymptomProbability('severe', 2, {'covid_incubation': 0.0,
                                         'covid_onset': 0.0,
                                         'covid_plateau': -1,
                                         'covid_post_plateau_1': -1,
                                         'covid_post_plateau_2': 0.0})
    ),
    (
        'extremely-severe',
        SymptomProbability('extremely-severe', 3, {'covid_incubation': 0.0,
                                                   'covid_onset': 0.0,
                                                   'covid_plateau': -1,
                                                   'covid_post_plateau_1': 0.0,
                                                   'covid_post_plateau_2': 0.0})
    ),

    # Symptoms

    (
        'fever',
        SymptomProbability('fever', 4, {'covid_incubation': 0.0,
                                        'covid_onset': 0.2,
                                        'covid_plateau': 0.8,
                                        'covid_post_plateau_1': 0.0,
                                        'covid_post_plateau_2': 0.0,
                                        'flu_first_day': 0.7,
                                        'flu': 0.7,
                                        'flu_last_day': 0.3})
    ),
    # 'fever' is a dependency of 'chills' so it needs to be inserted before
    # this position
    (
        'chills',
        SymptomProbability('chills', 5, {'covid_incubation': 0.0,
                                         'covid_onset': 0.8,
                                         'covid_plateau': 0.5,
                                         'covid_post_plateau_1': 0.0,
                                         'covid_post_plateau_2': 0.0})
    ),

    (
        'gastro',
        SymptomProbability('gastro', 6, {'covid_incubation': 0.0,
                                         'covid_onset': -1,
                                         'covid_plateau': -1,
                                         'covid_post_plateau_1': -1,
                                         'covid_post_plateau_2': -1,
                                         'flu_first_day': 0.7,
                                         'flu': 0.7,
                                         'flu_last_day': 0.2})
    ),
    # 'gastro' is a dependency of 'diarrhea' so it needs to be inserted before
    # this position
    (
        'diarrhea',
        SymptomProbability('diarrhea', 7, {'covid_incubation': 0.0,
                                           'covid_onset': 0.9,
                                           'covid_plateau': 0.9,
                                           'covid_post_plateau_1': 0.9,
                                           'covid_post_plateau_2': 0.9,
                                           'flu_first_day': 0.5,
                                           'flu': 0.5,
                                           'flu_last_day': 0.5})
    ),
    # 'gastro' is a dependency of 'nausea_vomiting' so it needs to be inserted
    # before this position
    (
        'nausea_vomiting',
        SymptomProbability('nausea_vomiting', 8, {'covid_incubation': 0.0,
                                                  'covid_onset': 0.7,
                                                  'covid_plateau': 0.7,
                                                  'covid_post_plateau_1': 0.7,
                                                  'covid_post_plateau_2': 0.7,
                                                  'flu_first_day': 0.5,
                                                  'flu': 0.5,
                                                  'flu_last_day': 0.25})
    ),

    # Age based lethargies
    # 'gastro' is a dependency of so it needs to be inserted before this
    # position
    (
        'fatigue',
        SymptomProbability('fatigue', 9, {'covid_incubation': 0.0,
                                          'covid_onset': -1,
                                          'covid_plateau': -1,
                                          'covid_post_plateau_1': -1,
                                          'covid_post_plateau_2': -1,
                                          'allergy': 0.2,
                                          'cold': 0.8,
                                          'cold_last_day': 0.8,
                                          'flu_first_day': 0.4,
                                          'flu': 0.8,
                                          'flu_last_day': 0.8})
    ),
    (
        'unusual',
        SymptomProbability('unusual', 10, {'covid_incubation': 0.0,
                                           'covid_onset': 0.2,
                                           'covid_plateau': 0.5,
                                           'covid_post_plateau_1': 0.5,
                                           'covid_post_plateau_2': 0.5})
    ),
    (
        'hard_time_waking_up',
        SymptomProbability('hard_time_waking_up', 11, {'covid_incubation': 0.0,
                                                       'covid_onset': 0.6,
                                                       'covid_plateau': 0.6,
                                                       'covid_post_plateau_1': 0.6,
                                                       'covid_post_plateau_2': 0.6,
                                                       'allergy': 0.3,
                                                       'flu_first_day': 0.3,
                                                       'flu': 0.5,
                                                       'flu_last_day': 0.4})
    ),
    (
        'headache',
        SymptomProbability('headache', 12, {'covid_incubation': 0.0,
                                            'covid_onset': 0.5,
                                            'covid_plateau': 0.5,
                                            'covid_post_plateau_1': 0.5,
                                            'covid_post_plateau_2': 0.5,
                                            'allergy': 0.6})
    ),
    (
        'confused',
        SymptomProbability('confused', 13, {'covid_incubation': 0.0,
                                            'covid_onset': 0.1,
                                            'covid_plateau': 0.1,
                                            'covid_post_plateau_1': 0.1,
                                            'covid_post_plateau_2': 0.1})
    ),
    (
        'lost_consciousness',
        SymptomProbability('lost_consciousness', 14, {'covid_incubation': 0.0,
                                                      'covid_onset': 0.1,
                                                      'covid_plateau': 0.1,
                                                      'covid_post_plateau_1': 0.1,
                                                      'covid_post_plateau_2': 0.1})
    ),

    # Respiratory symptoms
    # 'trouble_breathing' is a dependency of all this category so it should be
    # inserted before them
    (
        'trouble_breathing',
        SymptomProbability('trouble_breathing', 15, {'covid_incubation': 0.0,
                                                     'covid_onset': -1,
                                                     'covid_plateau': -1,
                                                     'covid_post_plateau_1': -1,
                                                     'covid_post_plateau_2': -1})
    ),
    (
        'sneezing',
        SymptomProbability('sneezing', 16, {'covid_incubation': 0.0,
                                            'covid_onset': 0.2,
                                            'covid_plateau': 0.3,
                                            'covid_post_plateau_1': 0.3,
                                            'covid_post_plateau_2': 0.3,
                                            'allergy': 1.0,
                                            'cold': 0.4,
                                            'cold_last_day': 0.0})
    ),
    (
        'cough',
        SymptomProbability('cough', 17, {'covid_incubation': 0.0,
                                         'covid_onset': 0.6,
                                         'covid_plateau': 0.9,
                                         'covid_post_plateau_1': 0.9,
                                         'covid_post_plateau_2': 0.9,
                                         'cold': 0.8,
                                         'cold_last_day': 0.8})
    ),
    (
        'runny_nose',
        SymptomProbability('runny_nose', 18, {'covid_incubation': 0.0,
                                              'covid_onset': 0.1,
                                              'covid_plateau': 0.2,
                                              'covid_post_plateau_1': 0.2,
                                              'covid_post_plateau_2': 0.2,
                                              'cold': 0.8,
                                              'cold_last_day': 0.8})
    ),
    (
        'sore_throat',
        SymptomProbability('sore_throat', 20, {'covid_incubation': 0.0,
                                               'covid_onset': 0.5,
                                               'covid_plateau': 0.8,
                                               'covid_post_plateau_1': 0.8,
                                               'covid_post_plateau_2': 0.8,
                                               'allergy': 0.3,
                                               'cold': 0.0,
                                               'cold_last_day': 0.6})
    ),
    (
        'severe_chest_pain',
        SymptomProbability('severe_chest_pain', 21, {'covid_incubation': 0.0,
                                                     'covid_onset': 0.4,
                                                     'covid_plateau': 0.5,
                                                     'covid_post_plateau_1': 0.15,
                                                     'covid_post_plateau_2': 0.15})
    ),

    # 'trouble_breathing' is a dependency of any '*_trouble_breathing' so it
    # needs to be inserted before this position
    (
        'light_trouble_breathing',
        SymptomProbability('light_trouble_breathing', 24, {'covid_incubation': 0.0,
                                                           'covid_onset': -1,
                                                           'covid_plateau': -1,
                                                           'covid_post_plateau_1': -1,
                                                           'covid_post_plateau_2': -1,
                                                           'allergy': 0.02})
    ),
    # This symptoms was in fact a mislabeled light_trouble_breathing
    (
        'mild_trouble_breathing',
        SymptomProbability('mild_trouble_breathing', 23, {})
    ),
    (
        'moderate_trouble_breathing',
        SymptomProbability('moderate_trouble_breathing', 25, {'covid_incubation': 0.0,
                                                              'covid_onset': -1,
                                                              'covid_plateau': -1,
                                                              'covid_post_plateau_1': -1,
                                                              'covid_post_plateau_2': -1})
    ),
    (
        'heavy_trouble_breathing',
        SymptomProbability('heavy_trouble_breathing', 26, {'covid_incubation': 0.0,
                                                           'covid_onset': 0,
                                                           'covid_plateau': -1,
                                                           'covid_post_plateau_1': -1,
                                                           'covid_post_plateau_2': -1})
    ),

    (
        'loss_of_taste',
        SymptomProbability('loss_of_taste', 22, {'covid_incubation': 0.0,
                                                 'covid_onset': 0.25,
                                                 'covid_plateau': 0.35,
                                                 'covid_post_plateau_1': 0.0,
                                                 'covid_post_plateau_2': 0.0})
    ),

    (
        'aches',
        SymptomProbability('aches', 19, {'flu_first_day': 0.3,
                                         'flu': 0.5,
                                         'flu_last_day': 0.8})
    )

    # commented out because these are not used elsewhere for now
    # (
    #     'hives',
    #     SymptomProbability('hives', __, {'allergy': 0.4})
    # ),
    # (
    #     'swelling',
    #     SymptomProbability('swelling', __, {'allergy': 0.3})
    # )
])


# 2D Array of symptoms; first axis is days after exposure (infection), second is an array of symptoms
def _get_covid_progression(initial_viral_load, viral_load_plateau_start, viral_load_plateau_end,
                           recovery_days, age, incubation_days, infectiousness_onset_days,
                           really_sick, extremely_sick, rng, preexisting_conditions, carefulness):
    """
    [summary]

    Args:
        initial_viral_load ([type]): [description]
        viral_load_plateau_start ([type]): [description]
        viral_load_plateau_end ([type]): [description]
        recovery_days (float): time to recover
        age ([type]): [description]
        incubation_days ([type]): [description]
        really_sick ([type]): [description]
        extremely_sick ([type]): [description]
        rng ([type]): [description]
        preexisting_conditions ([type]): [description]
        carefulness ([type]): [description]

    Returns:
        [type]: [description]
    """
    symptoms_contexts = SYMPTOMS_CONTEXTS['covid']
    progression = []
    symptoms_per_phase = [[] for i in range(len(symptoms_contexts))]

    # Phase 0 - Before onset of symptoms (incubation)
    # ====================================================
    phase_i = 0
    symptoms_per_phase[phase_i]= []
    # for day in range(math.ceil(incubation_days)):
    #     progression.append([])

    # Phase 1 of plateau
    # ====================================================
    phase_i = 1
    phase = symptoms_contexts[phase_i]

    p_fever = SYMPTOMS['fever'].probabilities[phase]
    if really_sick or extremely_sick or len(preexisting_conditions)>2 or initial_viral_load > 0.6:
        symptoms_per_phase[phase_i].append('moderate')
        p_fever = 0.4
    else:
        symptoms_per_phase[phase_i].append('mild')

    if rng.rand() < p_fever:
        symptoms_per_phase[phase_i].append('fever')
        if extremely_sick:
            if rng.rand() < 0.8:
                symptoms_per_phase[phase_i].append('chills')

    # gastro symptoms are more likely to be earlier and are more
    # likely to show extreme symptoms later
    p_gastro = initial_viral_load - .15
    if rng.rand() < p_gastro:
        symptoms_per_phase[phase_i].append('gastro')

        for symptom in ('diarrhea', 'nausea_vomiting'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    # fatigue and unusual symptoms are more heavily age-related
    # but more likely later, and less if you're careful/taking care
    # of yourself
    p_lethargy = (age/200) + initial_viral_load*0.6 - carefulness/2
    if rng.rand() < p_lethargy:
        symptoms_per_phase[phase_i].append('fatigue')

        if age > 75 and rng.rand() < SYMPTOMS['unusual'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('unusual')
        if really_sick or extremely_sick or len(preexisting_conditions) > 2 and \
                rng.rand() < SYMPTOMS['lost_consciousness'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('lost_consciousness')

        for symptom in ('hard_time_waking_up', 'headache', 'confused'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    # respiratory symptoms not so common at this stage
    p_respiratory = (0.5 * initial_viral_load) - (carefulness * 0.25) # e.g. 0.5*0.5 - 0.7*0.25 = 0.25-0.17
    if 'smoker' in preexisting_conditions or 'lung_disease' in preexisting_conditions:
        p_respiratory = (p_respiratory * 4) + age/200  # e.g. 0.1 * 4 * 45/200 = 0.4 + 0.225
    if rng.rand() < p_respiratory:
        symptoms_per_phase[phase_i].append('trouble_breathing')

        if extremely_sick and rng.rand() < SYMPTOMS['severe_chest_pain'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('severe_chest_pain')

        for symptom in ('sneezing', 'cough', 'runny_nose', 'sore_throat'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    if rng.rand() < SYMPTOMS['loss_of_taste'].probabilities[phase]:
        symptoms_per_phase[phase_i].append('loss_of_taste')

    if 'mild' in symptoms_per_phase[phase_i] and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('light_trouble_breathing')
    if 'moderate' in symptoms_per_phase[phase_i] and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('moderate_trouble_breathing')


    # During the symptoms plateau Part 2 (worst part of the disease)
    # ====================================================
    phase_i = 2
    phase = symptoms_contexts[phase_i]

    if extremely_sick:
        symptoms_per_phase[phase_i].append('extremely-severe')
    elif really_sick or len(preexisting_conditions) >2 or 'severe' in symptoms_per_phase[phase_i-1] or \
            initial_viral_load > 0.6:
        symptoms_per_phase[phase_i].append('severe')
    elif rng.rand() < p_gastro:
        symptoms_per_phase[phase_i].append('moderate')
    else:
        symptoms_per_phase[phase_i].append('mild')

    if 'fever' in symptoms_per_phase[phase_i-1] or initial_viral_load > 0.6 or \
            rng.rand() < SYMPTOMS['fever'].probabilities[phase]:
        symptoms_per_phase[phase_i].append('fever')
        if rng.rand() < SYMPTOMS['chills'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('chills')


    # gastro symptoms are more likely to be earlier and are more
    # likely to show extreme symptoms later (p gastro reduced compared to part1)
    if 'gastro' in symptoms_per_phase[phase_i-1] or rng.rand() < p_gastro *.25:
        symptoms_per_phase[phase_i].append('gastro')

        for symptom in ('diarrhea', 'nausea_vomiting'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    # fatigue and unusual symptoms are more heavily age-related
    # but more likely later, and less if you're careful/taking care
    # of yourself
    if rng.rand() < (p_lethargy + p_gastro): #if you had gastro symptoms before more likely to be lethargic now
        symptoms_per_phase[phase_i].append('fatigue')

        if age > 75 and rng.rand() < SYMPTOMS['unusual'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('unusual')
        if really_sick or extremely_sick or len(preexisting_conditions) > 2 and \
                rng.rand() < SYMPTOMS['lost_consciousness'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('lost_consciousness')

        for symptom in ('hard_time_waking_up', 'headache', 'confused'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    # respiratory symptoms more common at this stage
    p_respiratory = 2*(initial_viral_load - (carefulness * 0.25)) # e.g. 2* (0.5 - 0.7*0.25) = 2*(0.5-0.17)
    if 'smoker' in preexisting_conditions or 'lung_disease' in preexisting_conditions:
        p_respiratory = (p_respiratory * 4) + age/200  # e.g. 0.1 * 4 * 45/200 = 0.4 + 0.225
    if rng.rand() < p_respiratory:
        symptoms_per_phase[phase_i].append('trouble_breathing')

        if extremely_sick and rng.rand() < SYMPTOMS['severe_chest_pain'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('severe_chest_pain')

        for symptom in ('sneezing', 'cough', 'runny_nose', 'sore_throat'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    if 'mild' in symptoms_per_phase[phase_i] and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('light_trouble_breathing')
    if 'moderate' in symptoms_per_phase[phase_i] and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('moderate_trouble_breathing')
    if ('severe' in symptoms_per_phase[phase_i] or 'extremely-severe' in symptoms_per_phase[phase_i]) and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('heavy_trouble_breathing')

    if 'loss_of_taste' in symptoms_per_phase[phase_i-1] or \
            rng.rand() < SYMPTOMS['loss_of_taste'].probabilities[phase]:
        symptoms_per_phase[phase_i].append('loss_of_taste')


    # After the plateau (recovery part 1)
    # ====================================================
    phase_i = 3
    phase = symptoms_contexts[phase_i]

    if extremely_sick:
        symptoms_per_phase[phase_i].append('severe')
    elif really_sick:
        symptoms_per_phase[phase_i].append('moderate')
    else:
        symptoms_per_phase[phase_i].append('mild')

    # gastro symptoms are more likely to be earlier and are more
    # likely to show extreme symptoms later (p gastro reduced compared to part1)
    if 'gastro' in symptoms_per_phase[phase_i-1] or rng.rand() < p_gastro *.1:
        symptoms_per_phase[phase_i].append('gastro')

        for symptom in ('diarrhea', 'nausea_vomiting'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    # fatigue and unusual symptoms are more heavily age-related
    # but more likely later, and less if you're careful/taking care
    # of yourself
    if rng.rand() < (p_lethargy*1.5 + p_gastro): #if you had gastro symptoms before more likely to be lethargic now
        symptoms_per_phase[phase_i].append('fatigue')

        if age > 75 and rng.rand() < SYMPTOMS['unusual'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('unusual')
        if really_sick or extremely_sick or len(preexisting_conditions) > 2 and \
                rng.rand() < SYMPTOMS['lost_consciousness'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('lost_consciousness')

        for symptom in ('hard_time_waking_up', 'headache', 'confused'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    # respiratory symptoms more common at this stage but less than plateau
    p_respiratory = (initial_viral_load - (carefulness * 0.25)) # e.g. 2* (0.5 - 0.7*0.25) = 2*(0.5-0.17)
    if 'smoker' in preexisting_conditions or 'lung_disease' in preexisting_conditions:
        p_respiratory = (p_respiratory * 4) + age/200  # e.g. 0.1 * 4 * 45/200 = 0.4 + 0.225
    if rng.rand() < p_respiratory:
        symptoms_per_phase[phase_i].append('trouble_breathing')

        if extremely_sick and rng.rand() < SYMPTOMS['severe_chest_pain'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('severe_chest_pain')

        for symptom in ('sneezing', 'cough', 'runny_nose', 'sore_throat'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    if 'mild' in symptoms_per_phase[phase_i] and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('light_trouble_breathing')
    if 'moderate' in symptoms_per_phase[phase_i] and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('moderate_trouble_breathing')
    if ('severe' in symptoms_per_phase[phase_i] or 'extremely-severe' in symptoms_per_phase[phase_i]) and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('heavy_trouble_breathing')


    # After the plateau (recovery part 2)
    # ====================================================
    phase_i = 4
    phase = symptoms_contexts[phase_i]

    if extremely_sick:
        symptoms_per_phase[phase_i].append('moderate')
    else:
        symptoms_per_phase[phase_i].append('mild')

    # gastro symptoms are more likely to be earlier and are more
    # likely to show extreme symptoms later (p gastro reduced compared to part1)
    if 'gastro' in symptoms_per_phase[phase_i-1] or rng.rand() < p_gastro *.1:
        symptoms_per_phase[phase_i].append('gastro')

        for symptom in ('diarrhea', 'nausea_vomiting'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    # fatigue and unusual symptoms are more heavily age-related
    # but more likely later, and less if you're careful/taking care
    # of yourself
    if rng.rand() < (p_lethargy*2 + p_gastro): #if you had gastro symptoms before more likely to be lethargic now
        symptoms_per_phase[phase_i].append('fatigue')

        if age > 75 and rng.rand() < SYMPTOMS['unusual'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('unusual')
        if really_sick or extremely_sick or len(preexisting_conditions) > 2 and \
                rng.rand() < SYMPTOMS['lost_consciousness'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('lost_consciousness')

        for symptom in ('hard_time_waking_up', 'headache', 'confused'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    # respiratory symptoms getting less common
    p_respiratory = 0.5 * (initial_viral_load - (carefulness * 0.25)) # e.g. (0.5 - 0.7*0.25) = 0.5*(0.5-0.17)
    if 'smoker' in preexisting_conditions or 'lung_disease' in preexisting_conditions:
        p_respiratory = (p_respiratory * 4) + age/200  # e.g. 0.1 * 4 * 45/200 = 0.4 + 0.225
    if rng.rand() < p_respiratory:
        symptoms_per_phase[phase_i].append('trouble_breathing')

        if extremely_sick and rng.rand() < SYMPTOMS['severe_chest_pain'].probabilities[phase]:
            symptoms_per_phase[phase_i].append('severe_chest_pain')

        for symptom in ('sneezing', 'cough', 'runny_nose', 'sore_throat'):
            rand = rng.rand()
            if rand < SYMPTOMS[symptom].probabilities[phase]:
                symptoms_per_phase[phase_i].append(symptom)

    if 'mild' in symptoms_per_phase[phase_i] and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('light_trouble_breathing')
    if 'moderate' in symptoms_per_phase[phase_i] and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('moderate_trouble_breathing')
    if ('severe' in symptoms_per_phase[phase_i] or 'extremely-severe' in symptoms_per_phase[phase_i]) and \
            'trouble_breathing' in symptoms_per_phase[phase_i]:
        symptoms_per_phase[phase_i].append('heavy_trouble_breathing')

    viral_load_plateau_duration = math.ceil(viral_load_plateau_end - viral_load_plateau_start)
    recovery_duration = math.ceil(recovery_days - viral_load_plateau_end)
    incubation_days_wrt_infectiousness_onset_days = incubation_days - infectiousness_onset_days

    # same delay in symptom plateau as there was in symptom onset
    incubation_duration = math.ceil(incubation_days)
    covid_onset_duration = math.ceil(viral_load_plateau_start -
                                     incubation_days_wrt_infectiousness_onset_days) + \
                           viral_load_plateau_duration // 3
    plateau_duration = math.ceil(viral_load_plateau_duration * 2/3)
    post_plateau_1_duration = recovery_duration // 2
    post_plateau_2_duration = recovery_duration - post_plateau_1_duration

    assert viral_load_plateau_start >= incubation_days_wrt_infectiousness_onset_days

    for duration, symptoms in zip((incubation_duration, covid_onset_duration, plateau_duration,
                                   post_plateau_1_duration, post_plateau_2_duration),
                                  symptoms_per_phase):
        for day in range(duration):
            progression.append(symptoms)

    return progression


def _get_allergy_progression(rng):
    """
    [summary]

    Args:
        rng ([type]): [description]

    Returns:
        [type]: [description]
    """
    symptoms_contexts = SYMPTOMS_CONTEXTS['allergy']
    phase_i = 0
    phase = symptoms_contexts[phase_i]

    symptoms = []
    for symptom in ('sneezing', 'light_trouble_breathing', 'sore_throat', 'fatigue',
                    'hard_time_waking_up', 'headache'):
        rand = rng.rand()
        if rand < SYMPTOMS[symptom].probabilities[phase]:
            symptoms.append(symptom)

            # commented out because these are not used elsewhere for now
            # if symptom == 'light_trouble_breathing':
            #     for symptom in ('hives', 'swelling'):
            #         rand = rng.rand()
            #         if rand < SYMPTOMS[symptom].probabilities[phase]:
            #             symptoms.append(symptom)
    progression = [symptoms]
    return progression

def _get_flu_progression(age, rng, carefulness, preexisting_conditions, really_sick, extremely_sick, AVG_FLU_DURATION):
    """
    [summary]

    Args:
        age ([type]): [description]
        rng ([type]): [description]
        carefulness ([type]): [description]
        preexisting_conditions ([type]): [description]
        really_sick ([type]): [description]
        extremely_sick ([type]): [description]

    Returns:
        [type]: [description]
    """
    symptoms_contexts = SYMPTOMS_CONTEXTS['flu']
    symptoms_per_phase = [[] for _ in range(len(symptoms_contexts))]

    # Day 1 symptoms:
    phase_i = 0
    phase = symptoms_contexts[phase_i]

    symptoms_per_phase[phase_i].append('mild')

    for symptom in ('fatigue', 'fever', 'aches', 'hard_time_waking_up', 'gastro'):
        rand = rng.rand()
        if rand < SYMPTOMS[symptom].probabilities[phase]:
            symptoms_per_phase[phase_i].append(symptom)

            if symptom == 'gastro':
                for symptom in ('diarrhea', 'nausea_vomiting'):
                    rand = rng.rand()
                    if rand < SYMPTOMS[symptom].probabilities[phase]:
                        symptoms_per_phase[phase_i].append(symptom)

    # Day 2-4ish if it's a longer flu, if 2 days long this doesn't get added
    phase_i = 1
    phase = symptoms_contexts[phase_i]

    if really_sick or extremely_sick or any(preexisting_conditions):
        symptoms_per_phase[phase_i].append('moderate')
    else:
        symptoms_per_phase[phase_i].append('mild')

    for symptom in ('fatigue', 'fever', 'aches', 'hard_time_waking_up', 'gastro'):
        rand = rng.rand()
        if rand < SYMPTOMS[symptom].probabilities[phase]:
            symptoms_per_phase[phase_i].append(symptom)

            if symptom == 'gastro':
                for symptom in ('diarrhea', 'nausea_vomiting'):
                    rand = rng.rand()
                    if rand < SYMPTOMS[symptom].probabilities[phase]:
                        symptoms_per_phase[phase_i].append(symptom)

    # Last day
    phase_i = 2
    phase = symptoms_contexts[phase_i]

    symptoms_per_phase[phase_i].append('mild')

    for symptom in ('fatigue', 'fever', 'aches', 'hard_time_waking_up', 'gastro'):
        rand = rng.rand()
        if rand < SYMPTOMS[symptom].probabilities[phase]:
            symptoms_per_phase[phase_i].append(symptom)

            if symptom == 'gastro':
                for symptom in ('diarrhea', 'nausea_vomiting'):
                    rand = rng.rand()
                    if rand < SYMPTOMS[symptom].probabilities[phase]:
                        symptoms_per_phase[phase_i].append(symptom)

    if age < 12 or age > 40 or any(preexisting_conditions) or really_sick or extremely_sick:
        mean = AVG_FLU_DURATION + 2 - 2 * carefulness
    else:
        mean = AVG_FLU_DURATION - 2 * carefulness

    len_flu = rng.normal(mean,3)

    if len_flu < 2:
        len_flu = 3
    else:
        len_flu = round(len_flu)

    progression = []
    for duration, symptoms in zip((1, len_flu - 2, 1),
                                  symptoms_per_phase):
        for day in range(duration):
            progression.append(symptoms)

    return progression


def _get_cold_progression(age, rng, carefulness, preexisting_conditions, really_sick, extremely_sick):
    """
    [summary]

    Args:
        age ([type]): [description]
        rng ([type]): [description]
        carefulness ([type]): [description]
        preexisting_conditions ([type]): [description]
        really_sick ([type]): [description]
        extremely_sick ([type]): [description]

    Returns:
        [type]: [description]
    """
    symptoms_contexts = SYMPTOMS_CONTEXTS['cold']

    symptoms_per_phase = [[] for _ in range(len(symptoms_contexts))]

    # Day 2-4ish if it's a longer cold, if 2 days long this doesn't get added
    phase_i = 0
    phase = symptoms_contexts[phase_i]

    if really_sick or extremely_sick or any(preexisting_conditions):
        symptoms_per_phase[phase_i].append('moderate')
    else:
        symptoms_per_phase[phase_i].append('mild')

    for symptom in ('runny_nose', 'cough', 'fatigue', 'sneezing'):
        rand = rng.rand()
        if rand < SYMPTOMS[symptom].probabilities[phase]:
            symptoms_per_phase[phase_i].append(symptom)

    # Last day
    phase_i = 1
    phase = symptoms_contexts[phase_i]

    symptoms_per_phase[phase_i].append('mild')

    for symptom in ('runny_nose', 'cough', 'fatigue', 'sore_throat'):
        rand = rng.rand()
        if rand < SYMPTOMS[symptom].probabilities[phase]:
            symptoms_per_phase[phase_i].append(symptom)

    if age < 12 or age > 40 or any(preexisting_conditions) or really_sick or extremely_sick:
        mean = 4 - round(carefulness)
    else:
        mean = 3 - round(carefulness)

    len_cold = rng.normal(mean,3)
    if len_cold < 1:
        len_cold = 1
    else:
        len_cold = math.ceil(len_cold)

    progression = [[]]
    for duration, symptoms in zip((len_cold - 1, 1),
                                  symptoms_per_phase):
        for day in range(duration):
            progression.append(symptoms)

    return progression

def _reported_symptoms(all_symptoms, rng, carefulness):
    all_reported_symptoms = []
    for symptoms in all_symptoms:
        reported_symptoms = []
        # miss a day of symptoms
        if rng.rand() < carefulness:
            continue
        for symptom in symptoms:
            if rng.rand() < carefulness:
                continue
            reported_symptoms.append(symptom)
        all_reported_symptoms.append(reported_symptoms)
    return all_reported_symptoms
