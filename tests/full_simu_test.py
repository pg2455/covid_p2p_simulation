import datetime
import filecmp
import hashlib
import pickle
import unittest
import zipfile
from tempfile import NamedTemporaryFile

from run import run_simu
from simulator import Event


class FullUnitTest(unittest.TestCase):

    def test_simu_run(self):
        """
            run one simulation and ensure json files are correctly populated and most of the users have activity
        """
        with NamedTemporaryFile() as f:
            n_people = 100
            monitors = run_simu(
                n_stores=2,
                n_people=n_people,
                n_parks=1,
                n_hospitals=1,
                n_misc=2,
                init_percent_sick=0.1,
                start_time=datetime.datetime(2020, 2, 28, 0, 0),
                simulation_days=30,
                outfile=f.name,
                out_chunk_size=500
            )
            monitors[0].dump()
            monitors[0].join_iothread()
            f.seek(0)

            # Ensure
            data = []
            with zipfile.ZipFile(f"{f.name}.zip", 'r') as zf:
                for pkl in zf.namelist():
                    with zf.open(pkl) as pkl_f:
                        data.extend(pickle.load(pkl_f))

            self.assertTrue(len(data) > 0)

            self.assertTrue(Event.encounter in {d['event_type'] for d in data})
            self.assertTrue(Event.test in {d['event_type'] for d in data})

            self.assertTrue(len({d['human_id'] for d in data}) > n_people / 2)


class SeedUnitTest(unittest.TestCase):

    def setUp(self):
        self.test_seed = 136
        self.n_stores = 2
        self.n_people = 100
        self.n_parks = 1
        self.n_misc = 2
        self.n_hospitals = 2
        self.init_percent_sick = 0.1
        self.start_time = datetime.datetime(2020, 2, 28, 0, 0)
        self.simulation_days = 30

    def test_sim_same_seed(self):
        """
        Run two simulations with the same seed and ensure we get the same output
        Note: If this test is failing, it is a good idea to load the data of both files and use DeepDiff to compare
        """
        with NamedTemporaryFile() as f1, NamedTemporaryFile() as f2, NamedTemporaryFile() as f3:
            monitors1 = run_simu(
                n_stores=self.n_stores,
                n_people=self.n_people,
                n_parks=self.n_parks,
                n_misc=self.n_misc,
                n_hospitals=self.n_hospitals,
                init_percent_sick=self.init_percent_sick,
                start_time=self.start_time,
                simulation_days=self.simulation_days,
                outfile=f1.name,
                out_chunk_size=0,
                out_humans=f3.name,
                seed=self.test_seed
            )
            monitors1[0].dump()
            monitors1[0].join_iothread()
            f1.seek(0)

            monitors2 = run_simu(
                n_stores=self.n_stores,
                n_people=self.n_people,
                n_parks=self.n_parks,
                n_misc=self.n_misc,
                n_hospitals=self.n_hospitals,
                init_percent_sick=self.init_percent_sick,
                start_time=self.start_time,
                simulation_days=self.simulation_days,
                outfile=f2.name,
                out_chunk_size=0,
                out_humans=f3.name,
                seed=self.test_seed
            )
            monitors2[0].dump()
            monitors2[0].join_iothread()
            f2.seek(0)

            md5 = hashlib.md5()
            with zipfile.ZipFile(f"{f1.name}.zip", 'r') as zf:
                for pkl in zf.namelist():
                    with zf.open(pkl) as pkl_f:
                        md5.update(pkl_f.read())
            md5sum1 = md5.hexdigest()

            md5 = hashlib.md5()
            with zipfile.ZipFile(f"{f2.name}.zip", 'r') as zf:
                for pkl in zf.namelist():
                    with zf.open(pkl) as pkl_f:
                        md5.update(pkl_f.read())
            md5sum2 = md5.hexdigest()

            self.assertTrue(md5sum1 == md5sum2,
                            msg=f"Two simulations run with the same seed "
                            f"({self.test_seed}) yielded different results")

    def test_sim_diff_seed(self):
        """
        Using different seeds should yield different output
        """

        with NamedTemporaryFile() as f1, NamedTemporaryFile() as f2, NamedTemporaryFile() as f3:
            monitors1 = run_simu(
                n_stores=self.n_stores,
                n_people=self.n_people,
                n_parks=self.n_parks,
                n_misc=self.n_misc,
                init_percent_sick=self.init_percent_sick,
                start_time=self.start_time,
                simulation_days=self.simulation_days,
                outfile=f1.name,
                out_chunk_size=0,
                out_humans=f3.name,
                seed=self.test_seed
            )
            monitors1[0].dump()
            monitors1[0].join_iothread()
            f1.seek(0)

            monitors2 = run_simu(
                n_stores=self.n_stores,
                n_people=self.n_people,
                n_parks=self.n_parks,
                n_misc=self.n_misc,
                init_percent_sick=self.init_percent_sick,
                start_time=self.start_time,
                simulation_days=self.simulation_days,
                outfile=f2.name,
                out_chunk_size=0,
                out_humans=f3.name,
                seed=self.test_seed+1
            )
            monitors2[0].dump()
            monitors2[0].join_iothread()
            f2.seek(0)

            md5 = hashlib.md5()
            with zipfile.ZipFile(f"{f1.name}.zip", 'r') as zf:
                for pkl in zf.namelist():
                    with zf.open(pkl) as pkl_f:
                        md5.update(pkl_f.read())
            md5sum1 = md5.hexdigest()

            md5 = hashlib.md5()
            with zipfile.ZipFile(f"{f2.name}.zip", 'r') as zf:
                for pkl in zf.namelist():
                    with zf.open(pkl) as pkl_f:
                        md5.update(pkl_f.read())
            md5sum2 = md5.hexdigest()

            self.assertFalse(md5sum1 == md5sum2,
                             msg=f"Two simulations run with different seeds "
                             f"({self.test_seed},{self.test_seed+1}) yielded "
                             f"the same result")
