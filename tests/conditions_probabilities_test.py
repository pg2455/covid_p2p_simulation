import unittest

import numpy as np

from utils import _get_preexisting_conditions, PREEXISTING_CONDITIONS


class PreexistingConditions(unittest.TestCase):
    def test_preexisting_conditions(self):
        """
            Test the distribution of the preexisting conditions
        """
        n_people = 10000

        rng = np.random.RandomState(1234)

        for c_name, c_prob in PREEXISTING_CONDITIONS.items():
            for p in c_prob:
                computed_dist = [
                    # note: we currently need to disable modifiers for combined precondition
                    # (...the modifiers are pretty hacky and don't seem to fit w/ the marginals)
                    _get_preexisting_conditions(p.age-1, p.sex, rng, disable_modifiers=True)
                    for _ in range(n_people)]
                prob = len([cd for cd in computed_dist if p.name in cd]) / n_people
                self.assertTrue(p.probability == 0.0 or abs(p.probability - prob) < 0.01,
                                msg=f"Computation of the preexisting conditions for '{p.name}' "
                                    f"yielded a different probability than expected: {prob} "
                                    f"instead of {p.probability}")
