from typing import Dict


class Event:
    test = 'test'
    encounter = 'encounter'
    symptom_start = 'symptom_start'
    contamination = 'contamination'

    @staticmethod
    def members():
        return [Event.test, Event.encounter, Event.symptom_start, Event.contamination]

    @staticmethod
    def log_encounter(human1, human2, location, duration, distance, time):
        human1.events.append(
            {
                'time': time,
                'event_type': Event.encounter,
                'human_id': human1.name,
                'encounter': {
                    'time': time,
                    'event_type': Event.encounter,
                    'duration': duration,
                    'distance': distance,
                    'location_type': location.location_type,
                    'contamination_prob': location.cont_prob,
                    'lat': location.lat,
                    'lon': location.lon,
                    'obs_lat': human1.obs_lat,
                    'obs_lon': human1.obs_lon,
                },
                'human1': Event.dump_human(human1),
                'human2': Event.dump_human(human2, prefix="other_")
            }
        )

        human2.events.append(
            {
                'time': time,
                'event_type': Event.encounter,
                'human_id': human2.name,
                'encounter': {
                    'time': time,
                    'event_type': Event.encounter,
                    'duration': duration,
                    'distance': distance,
                    'location_type': location.location_type,
                    'contamination_prob': location.cont_prob,
                    'lat': location.lat,
                    'lon': location.lon,
                    'obs_lat': human2.obs_lat,
                    'obs_lon': human2.obs_lon,
                },
                'human1': Event.dump_human(human1, prefix="other_"),
                'human2': Event.dump_human(human2)
            }
        )

    @staticmethod
    def log_test(human, result, time):
        human.events.append(
            {
                'human_id': human.name,
                'event_type': Event.test,
                'time': time,
                'payload': {
                    'result': result,
                }
            }
        )

    @staticmethod
    def log_symptom_start(human, time, covid=True):
        human.events.append(
            {
                'human_id': human.name,
                'event_type': Event.symptom_start,
                'time': time,
                'payload': {
                    'covid': covid
                }
            }
        )

    @staticmethod
    def log_contaminate(human, time):
        human.events.append(
            {
                'human_id': human.name,
                'event_type': Event.contamination,
                'time': time,
                'payload': {}
            }
        )

    @staticmethod
    def dump_human(human, prefix="") -> Dict:
        return {
            f'{prefix}human_id': human.name,
            f'{prefix}age': human.age,
            f'{prefix}carefullness': human.carefullness,

            # To update to health as a non-obersable variable
            f'{prefix}health': human.health,
            f'{prefix}is_infected': human.health['is_infected'],
            f'{prefix}infection_timestamp': human.health['infection_timestamp'],
            f'{prefix}infectiousness': human.health['infectiousness'],
            f'{prefix}reported_symptoms': human.health['reported_symptoms'],
            f'{prefix}symptoms': human.health['symptoms'],
            f'{prefix}test_results': human.health['test_results'],
            f'{prefix}has_app': human.has_app
        }
