import datetime
import glob
import os
import pickle
import shutil
from tempfile import NamedTemporaryFile, TemporaryDirectory
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
    def test_run(self):
        """
            run one simulation and ensure json files are correctly populated and most of the users have activity
        """
        with NamedTemporaryFile(suffix='.zip') as logs_f, \
             TemporaryDirectory() as preprocess_d:
            n_people = 35
            monitors, _ = run_simu(
                n_people=n_people,
                init_percent_sick=0.25,
                start_time=datetime.datetime(2020, 2, 28, 0, 0),
                simulation_days=30,
                outfile=logs_f.name[:-len('.zip')],
                out_chunk_size=0,
                seed=0
            )
            monitors[0].dump()
            monitors[0].join_iothread()

            os.makedirs(os.path.join(preprocess_d, 'plots', 'risk'))
            os.makedirs(os.path.join(preprocess_d, 'daily_outputs'))
            cluster_data_input_path = os.path.join(preprocess_d, os.path.basename(logs_f.name))
            shutil.copy(logs_f.name, cluster_data_input_path)

            args = m_parser.parse_args([
                f'--plot_path={os.path.join(preprocess_d, "plots", "risk")}',
                f'--cluster_path={os.path.join(preprocess_d, "clusters.json")}',
                f'--output_file={os.path.join(preprocess_d, "output.pkl")}',
                f'--data_path={cluster_data_input_path}',
                '--risk_model=tristan',
                '--seed=0',
                '--save_training_data',
                '--n_jobs=4'
            ])

            m_main(args)

            days_output_prefix = f"{preprocess_d}/daily_outputs/"
            days_output = glob.glob(days_output_prefix + "*/")
            days_output = sorted(days_output, key=lambda path:
                    int(path[len(days_output_prefix):-1]))

            output = []
            for day_output in days_output:
                pkls = glob.glob(f"{day_output}*/daily_human.pkl")
                pkls.sort()
                day_humans = []
                for pkl in pkls:
                    with open(pkl, 'rb') as f:
                        day_humans.append(pickle.load(f))
                output.append(day_humans)

            stats = {'humans': {}}

            for current_day, day_output in enumerate(output):
                for h_i, human in enumerate(day_output):
                    stats['humans'].setdefault(h_i, {})
                    stats['humans'][h_i].setdefault('candidate_encounters_cnt', 0)
                    stats['humans'][h_i].setdefault('has_exposure_day', 0)
                    stats['humans'][h_i].setdefault('has_infectious_day', 0)
                    stats['humans'][h_i].setdefault('has_recovery_day', 0)
                    stats['humans'][h_i].setdefault('exposure_encounter_cnt', 0)

                    self.assertEqual(current_day, human['current_day'])

                    observed = human['observed']
                    unobserved = human['unobserved']

                    if current_day == 0:
                        prev_observed = None
                        prev_unobserved = None
                    else:
                        prev_observed = output[current_day - 1][h_i]['observed']
                        prev_unobserved = output[current_day - 1][h_i]['unobserved']

                    # Multi-hot arrays identifying the reported symptoms in the last 14 days
                    # Symptoms:
                    # ['aches', 'cough', 'fatigue', 'fever', 'gastro', 'loss_of_taste',
                    #  'mild', 'moderate', 'runny_nose', 'severe', 'trouble_breathing']
                    self.assertEqual(observed['reported_symptoms'].shape[0], 14)
                    if len(observed['candidate_encounters']):
                        stats['humans'][h_i]['candidate_encounters_cnt'] += 1
                        # according to models/helper.py (as of 2020/04/24):
                        # candidate_encounters[:, 0] is the cluster id of the encounter message (?)
                        # candidate_encounters[:, 1] is the risk of the first (??) message in the cluster
                        # candidate_encounters[:, 2] is the number of messages in the cluster
                        # candidate_encounters[:, 3] is the 'day' timestamp of the cluster (for slicing)
                        self.assertEqual(observed['candidate_encounters'].shape[1], 4)
                        self.assertGreaterEqual(observed['candidate_encounters'][:, 0].min(), 0)
                        #self.assertLess(observed['candidate_encounters'][:, 0].max(), 16)  # cluster id type = ??
                        self.assertGreaterEqual(observed['candidate_encounters'][:, 1].min(), 0)
                        self.assertLess(observed['candidate_encounters'][:, 1].max(), 16)  # 4-bit-risk max = 15
                        self.assertGreaterEqual(observed['candidate_encounters'][:, 2].min(), 0)
                        #self.assertLess(observed['candidate_encounters'][:, 2].max(), 14)  # max msg per cluster = ??
                        self.assertGreaterEqual(observed['candidate_encounters'][:, 3].min(), 0)
                        #self.assertLess(observed['candidate_encounters'][:, 3].max(), 14)  # max 14 days old (?)
                    # Has received a positive test result [index] days before today
                    self.assertEqual(observed['test_results'].shape, (14,))
                    self.assertTrue(observed['test_results'].min() in (0, 1))
                    self.assertTrue(observed['test_results'].max() in (0, 1))
                    self.assertTrue(observed['test_results'].sum() in (0, 1))

                    # Multi-hot arrays identifying the true symptoms in the last 14 days
                    # Symptoms:
                    # ['aches', 'cough', 'fatigue', 'fever', 'gastro', 'loss_of_taste',
                    #  'mild', 'moderate', 'runny_nose', 'severe', 'trouble_breathing']
                    self.assertTrue(unobserved['true_symptoms'].shape[0] == 14)
                    # Has been exposed or not
                    self.assertTrue(unobserved['is_exposed'] in (0, 1))
                    if unobserved['exposure_day'] is not None:
                        stats['humans'][h_i]['has_exposure_day'] = 1
                        # For how long has been exposed
                        self.assertTrue(0 <= unobserved['exposure_day'] < 14)
                    # Is infectious or not
                    self.assertTrue(unobserved['is_infectious'] in (0, 1))
                    if unobserved['infectious_day'] is not None:
                        stats['humans'][h_i]['has_infectious_day'] = 1
                        # For how long has been infectious
                        self.assertTrue(0 <= unobserved['infectious_day'] < 14)
                    # Is recovered or not
                    self.assertTrue(unobserved['is_recovered'] in (0, 1))
                    if unobserved['infectious_day'] is not None:
                        stats['humans'][h_i]['has_recovery_day'] = 1
                        # For how long has been infectious
                        if unobserved['recovery_day'] is not None:
                            self.assertTrue(0 <= unobserved['recovery_day'] < 14)
                    if len(observed['candidate_encounters']):
                        stats['humans'][h_i]['exposure_encounter_cnt'] += 1
                        # Encounters responsible for exposition. Exposition can occur without being
                        # linked to an encounter
                        self.assertTrue(len(unobserved['exposure_encounter'].shape) == 1)
                        self.assertTrue(unobserved['exposure_encounter'].min() in (0, 1))
                        self.assertTrue(unobserved['exposure_encounter'].max() in (0, 1))
                        self.assertTrue(unobserved['exposure_encounter'].sum() in (0, 1))
                    # Level of infectiousness / day
                    self.assertTrue(unobserved['infectiousness'].shape == (14,))
                    self.assertTrue(unobserved['infectiousness'].min() >= 0)
                    self.assertTrue(unobserved['infectiousness'].max() <= 1)

                    # observed['reported_symptoms'] is a subset of unobserved['true_symptoms']
                    self.assertTrue((unobserved['true_symptoms'] == observed['reported_symptoms'])
                                    [observed['reported_symptoms'].astype(np.bool)].all())

                    if unobserved['is_infectious'] or unobserved['is_recovered']:
                        self.assertTrue(unobserved['is_infectious'] != unobserved['is_recovered'])

                    # exposure_encounter is the same length as candidate_encounters
                    self.assertTrue(unobserved['exposure_encounter'].shape == (observed['candidate_encounters'].shape[0],))

                    if prev_observed:
                        self.assertTrue((observed['reported_symptoms'][:13] == prev_observed['reported_symptoms'][-13:]).all())
                        # broken in all kinds of ways
                        #self.assertTrue((observed['candidate_encounters']
                        #                 [observed['candidate_encounters'][:, 2] > 1][:, 0:2] ==
                        #                 prev_observed['candidate_encounters']
                        #                 [prev_observed['candidate_encounters'][:, 2] < 13][:, 0:2]).all())
                        #self.assertTrue((observed['test_results'][:13] == prev_observed['test_results'][-13:]).all())
                        self.assertTrue((unobserved['true_symptoms'][:13] == prev_unobserved['true_symptoms'][-13:]).all())
                        self.assertTrue(unobserved['is_exposed'] if prev_unobserved['is_exposed'] else True)
                        #self.assertTrue((unobserved['infectiousness'][:13] == prev_unobserved['infectiousness'][-13:]).all())
                        if prev_unobserved['exposure_day'] is not None:
                            self.assertTrue(min(0, unobserved['exposure_day'] + 1) == prev_unobserved['exposure_day'])

                        if unobserved['is_exposed'] != prev_unobserved['is_exposed']:
                            self.assertTrue(unobserved['is_exposed'])
                            self.assertTrue(unobserved['exposure_day'] == 0)
                            self.assertTrue(prev_unobserved['infectiousness'][0] == 0)