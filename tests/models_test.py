import datetime
import pickle
from tempfile import NamedTemporaryFile
import unittest

import numpy as np

from models.run import parser as m_parser, main as m_main
from run import run_simu

# Force COLLECT_LOGS=True
import config
from base import Event
import simulator
config.COLLECT_LOGS = True
simulator.Event = Event


class ModelsPreprocessingTest(unittest.TestCase):

    def test_simu_run(self):
        """
            run one simulation and ensure json files are correctly populated and most of the users have activity
        """
        with NamedTemporaryFile() as logs_f, \
             NamedTemporaryFile() as preprocess_f:
            n_people = 100
            monitors, _ = run_simu(
                n_people=n_people,
                init_percent_sick=0.1,
                start_time=datetime.datetime(2020, 2, 28, 0, 0),
                simulation_days=30,
                outfile=logs_f.name,
                out_chunk_size=0,
                seed=0
            )
            monitors[0].dump()
            monitors[0].join_iothread()
            logs_f.seek(0)

            args = m_parser.parse_args([f'--data_path={logs_f.name}.zip',
                                        f'--output_file={preprocess_f.name}',
                                        '--risk_model=tristan',
                                        '--seed=0', '--save_training_data',
                                        '--n_jobs=8'])
            m_main(args)
            preprocess_f.seek(0)

            output = pickle.load(preprocess_f.name)

            for current_day, daily_output in enumerate(output):
                assert current_day == daily_output['current_day']

                for h_i, (h_k, human) in enumerate(daily_output.items()):
                    observed = human['observed']
                    unobserved = human['unobserved']

                    try:
                        prev_observed = daily_output[current_day - 1][h_i][h_k]['observed']
                        prev_unobserved = daily_output[current_day - 1][h_i][h_k]['unobserved']
                    except IndexError:
                        prev_observed = None
                        prev_unobserved = None

                    # Multi-hot arrays identifying the reported symptoms in the last 14 days
                    # Symptoms:
                    # ['aches', 'cough', 'fatigue', 'fever', 'gastro', 'loss_of_taste',
                    #  'mild', 'moderate', 'runny_nose', 'severe', 'trouble_breathing']
                    assert observed['reported_symptoms'].shape == (14, 11)
                    # candidate_encounters[:, 0] is the other human 4 bits id
                    # candidate_encounters[:, 1] is the risk of getting contaminated during the encounter?
                    # candidate_encounters[:, 2] is the number of days since the encounter
                    assert observed['candidate_encounters'].shape[1] == 3
                    assert observed['candidate_encounters'][:, 0].min() >= 0
                    assert observed['candidate_encounters'][:, 0].max() < 16
                    assert observed['candidate_encounters'][:, 1].min() >= 0
                    assert observed['candidate_encounters'][:, 1].max() < 16
                    assert observed['candidate_encounters'][:, 2].min() >= 0
                    assert observed['candidate_encounters'][:, 2].max() < 14
                    # Has received a positive test result [index] days before today
                    assert unobserved['test_results'].shape == (14,)
                    assert unobserved['test_results'].min() in (0, 1)
                    assert unobserved['test_results'].max() in (0, 1)
                    assert unobserved['test_results'].sum() in (0, 1)

                    # Multi-hot arrays identifying the true symptoms in the last 14 days
                    # Symptoms:
                    # ['aches', 'cough', 'fatigue', 'fever', 'gastro', 'loss_of_taste',
                    #  'mild', 'moderate', 'runny_nose', 'severe', 'trouble_breathing']
                    assert unobserved['true_symptoms'].shape == (14, 11)
                    # Has been exposed or not
                    assert unobserved['is_exposed'] in (0, 1)
                    # For how long has been exposed
                    assert 0 <= unobserved['exposure_day'] < 14
                    # Is infectious or not
                    assert unobserved['is_infectious'] in (0, 1)
                    # For how long has been infectious
                    assert 0 <= unobserved['infectious_day'] < 14
                    # Is recovered or not
                    assert unobserved['is_recovered'] in (0, 1)
                    # For how long has been infectious
                    assert 0 <= unobserved['recovery_day'] < 14
                    # Locations where unobserved['is_exposed'] was true
                    assert len(unobserved['exposed_locs'].shape) == 1
                    assert unobserved['exposed_locs'].min() in (0, 1)
                    assert unobserved['exposed_locs'].max() in (0, 1)
                    assert 0 <= unobserved['exposed_locs'].sum() <= len(unobserved['exposed_locs'])
                    # Encounters responsible for exposition. Exposition can occur without being
                    # linked to an encounter
                    assert len(unobserved['exposure_encounter'].shape) == 1
                    assert unobserved['exposure_encounter'].min() in (0, 1)
                    assert unobserved['exposure_encounter'].max() in (0, 1)
                    assert unobserved['exposure_encounter'].sum() in (0, 1)
                    # Level of infectiousness / day
                    assert unobserved['infectiousness'].shape == (14,)
                    assert unobserved['infectiousness'].min() >= 0
                    assert unobserved['infectiousness'].max() <= 1

                    # observed['reported_symptoms'] is a subset of unobserved['true_symptoms']
                    assert (unobserved['true_symptoms'] == observed['reported_symptoms']) \
                           [observed['reported_symptoms'].astype(np.bool)].all()

                    if unobserved['is_infectious'] or unobserved['is_recovered']:
                        assert unobserved['is_infectious'] != unobserved['is_recovered']

                    # exposed_locs is the same length as candidate_locs
                    # TODO: observed['candidate_locs'] should be a tuple (human_readable, id) preferably sorted
                    assert unobserved['exposed_locs'].shape == (len(observed['candidate_locs']),)

                    # exposure_encounter is the same length as candidate_encounters
                    assert unobserved['exposure_encounter'].shape == (observed['candidate_encounters'].shape[0],)

                    if prev_observed:
                        assert (observed['reported_symptoms'][:13, :] == prev_observed['reported_symptoms'][-13:, :]).all()
                        assert (observed['candidate_encounters'][observed['candidate_encounters'][:, 2] > 1][:, 0:2] ==
                                prev_observed['candidate_encounters'][prev_observed['candidate_encounters'][:, 2] < 13][:, 0:2]).all()
                        assert (observed['test_results'][:13, :] == prev_observed['test_results'][-13:, :]).all()

                        assert (unobserved['true_symptoms'][:13, :] == prev_unobserved['true_symptoms'][-13:, :]).all()
                        assert unobserved['is_exposed'] if prev_unobserved['is_exposed'] else True
                        assert (unobserved['infectiousness'][:13, :] == prev_unobserved['infectiousness'][-13:, :]).all()

                        assert min(0, unobserved['exposure_day'] + 1) == prev_unobserved['exposure_day']

                        if unobserved['is_exposed'] != prev_unobserved['is_exposed']:
                            assert unobserved['is_exposed']
                            assert unobserved['exposure_day'] == 0
                            assert unobserved['exposed_locs'].sum() == prev_unobserved['exposed_locs'].sum() + 1
                            assert prev_unobserved['infectiousness'][0] == 0
