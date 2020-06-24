"""
[summary]
"""
import datetime
import math
import os
import pathlib
import subprocess
import typing
import zipfile
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import time
import dill
import numpy as np
import requests
import yaml
from omegaconf import DictConfig, OmegaConf
from scipy.stats import norm


def log(str, logfile=None, timestamp=False):
    """
    [summary]

    Args:
        str ([type]): [description]
        logfile ([type], optional): [description]. Defaults to None.
        timestamp (bool, optional): [description]. Defaults to False.
    """
    if timestamp:
        str = f"[{datetime.datetime.now()}] {str}"

    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)


def _normalize_scores(scores):
    """
    [summary]

    Args:
        scores ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.array(scores)/np.sum(scores)


def _get_random_area(num, total_area, rng):
    """
    Using Dirichlet distribution since it generates a "distribution of probabilities"
    which will ensure that the total area allotted to a location type remains conserved
    while also maintaining a uniform distribution

    Args:
        num ([type]): [description]
        total_area ([type]): [description]
        rng ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Keeping max at area/2 to ensure no location is allocated more than half of the total area allocated to its location type
    area = np.array([total_area/num for _ in range(num)])
    return area

def draw_random_discrete_gaussian(avg, scale, rng):
    """
    [summary]

    Args:
        avg ([type]): [description]
        scale ([type]): [description]
        rng ([type]): [description]

    Returns:
        [type]: [description]
    """
    # https://stackoverflow.com/a/37411711/3413239
    irange, normal_pdf = _get_integer_pdf(avg, scale, 2)
    return int(rng.choice(irange, size=1, p=normal_pdf))

def _json_serialize(o):
    """
    [summary]

    Args:
        o ([type]): [description]

    Returns:
        [type]: [description]
    """
    if isinstance(o, datetime.datetime):
        return o.__str__()

def compute_distance(loc1, loc2):
    """
    [summary]

    Args:
        loc1 ([type]): [description]
        loc2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.sqrt((loc1.lat - loc2.lat) ** 2 + (loc1.lon - loc2.lon) ** 2)


@lru_cache(500)
def _get_integer_pdf(avg, scale, num_sigmas=2):
    """
    [summary]

    Args:
        avg ([type]): [description]
        scale ([type]): [description]
        num_sigmas (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    irange = np.arange(avg - num_sigmas * scale, avg + num_sigmas * scale + 1)
    normal_pdf = norm.pdf(irange - avg)
    normal_pdf /= normal_pdf.sum()
    return irange, normal_pdf


def probas_to_risk_mapping(probas,
                           num_bins,
                           lower_cutoff=None,
                           upper_cutoff=None):
    """
    Create a mapping from probabilities returned by the model to discrete
    risk levels, with a number of predictions in each bins being approximately
    equivalent.

    Args:
        probas (np.ndarray): The array of probabilities returned by the model.
        num_bins (int): The number of bins. For example, `num_bins=16` for risk
            messages on 4 bits.
        lower_cutoff (float, optional): Ignore values smaller than `lower_cutoff`
            in the creation of the bins. This avoids any bias towards values which
            are too close to 0. If `None`, then do not cut off the small probabilities.
            Defaults to None.
        upper_cutoff (float, optional): Ignore values larger than `upper_cutoff` in the
            creation of the bins. This avoids any bias towards values which are too
            close to 1. If `None`, then do not cut off the large probabilities.
            Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        np.ndarray: The mapping from probabilities to discrete risk levels. This mapping has
        size `num_bins + 1`, with the first values always being 0, and the last
        always being 1.
    """
    if (lower_cutoff is not None) and (upper_cutoff is not None):
        if lower_cutoff >= upper_cutoff:
            raise ValueError('The lower cutoff must have a value which is '
                             'smaller than the upper cutoff, got `lower_cutoff='
                             '{0}` and `upper_cutoff={1}`.'.format(
                             lower_cutoff, upper_cutoff))
    mask = np.ones_like(probas, dtype=np.bool_)
    num_percentiles = num_bins + 1
    # First value is always 0, last value is always 1
    cutoffs = np.zeros((num_bins + 1,), dtype=probas.dtype)
    cutoffs[-1] = 1.

    # Remove probabilities close to 0
    lower_idx = 1 if (lower_cutoff is None) else None
    if lower_cutoff is not None:
        mask = np.logical_and(mask, probas > lower_cutoff)
        num_percentiles -= 1

    # Remove probabilities close to 1
    upper_idx = -1 if (upper_cutoff is None) else None
    if upper_cutoff is not None:
        mask = np.logical_and(mask, probas <= upper_cutoff)
        num_percentiles -= 1

    percentiles = np.linspace(0, 100, num_percentiles)
    cutoffs[1:-1] = np.percentile(probas[mask],
                                  q=percentiles[lower_idx:upper_idx])

    return cutoffs

def proba_to_risk_fn(mapping):
    """
    Create a callable, based on a mapping, that takes probabilities (in
    [0, 1]) and returns a discrete risk level (in [0, num_bins - 1]).

    Args:
        mapping (np.ndarray): The mapping from probabilities to discrete risk levels.
        See `probas_to_risk_mapping`.

    Returns:
        callable: Function taking probabilities and returning discrete risk levels.
    """
    def _proba_to_risk(probas):
        return np.maximum(np.searchsorted(mapping, probas, side='left') - 1, 0)

    return _proba_to_risk

def calculate_average_infectiousness(human):
    """ This is only used for the infectiousness value for a human that is written out for the ML predictor.
    We write tomorrows infectiousness (and predict tomorrows infectiousness) so that our predictor is conservative. """
    # cur_infectiousness = human.get_infectiousness_for_day(human.env.timestamp, human.is_infectious)
    is_infectious_tomorrow = True if human.infection_timestamp and human.env.timestamp - human.infection_timestamp + datetime.timedelta(days=1) >= datetime.timedelta(days=human.infectiousness_onset_days) else False
    tomorrows_infectiousness = human.get_infectiousness_for_day(human.env.timestamp + datetime.timedelta(days=1),
                                                                is_infectious_tomorrow)
    return tomorrows_infectiousness #(cur_infectiousness + tomorrows_infectiousness) / 2


def filter_open(locations):
    """Given an iterable of locations, returns a list of those that are open for business.

    Args:
        locations (iterable): a list of objects inheriting from the covid19sim.base.Location class

    Returns:
        list
    """
    return [loc for loc in locations if loc.is_open_for_business]

def filter_queue_max(locations, max_len):
    """Given an iterable of locations, will return a list of those
    with queues that are not too long.

    Args:
        locations (iterable): a list of objects inheriting from the covid19sim.base.Location class

    Returns:
        list
    """
    return [loc for loc in locations if len(loc.queue) <= max_len]


def download_file_from_google_drive(
        gdrive_file_id: typing.AnyStr,
        destination: typing.AnyStr,
        chunk_size: int = 32768
) -> typing.AnyStr:
    """
    Downloads a file from google drive, bypassing the confirmation prompt.

    Args:
        gdrive_file_id: ID string of the file to download from google drive.
        destination: where to save the file.
        chunk_size: chunk size for gradual downloads.

    Returns:
        The path to the downloaded file.
    """
    # taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': gdrive_file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        params = {'id': gdrive_file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    return destination


def download_exp_data_if_not_exist(
        exp_data_url: typing.AnyStr,
        exp_data_destination: typing.AnyStr,
) -> typing.AnyStr:
    """
    Downloads & extract config/weights for a model if the provided destination does not exist.

    The provided URL will be assumed to be a Google Drive download URL. The download will be
    skipped entirely if the destination folder already exists. This function will return the
    path to the existing folder, or to the newly created folder.

    Args:
        exp_data_url: the zip URL (under the `https://drive.google.com/file/d/ID` format).
        exp_data_destination: where to extract the model data.

    Returns:
        The path to the model data.
    """
    assert exp_data_url.startswith("https://drive.google.com/file/d/")
    gdrive_file_id = exp_data_url.split("/")[-1]
    output_data_path = os.path.join(exp_data_destination, gdrive_file_id)
    downloaded_zip_path = os.path.join(exp_data_destination, f"{gdrive_file_id}.zip")
    if os.path.isfile(downloaded_zip_path) and os.path.isdir(output_data_path):
        return output_data_path
    os.makedirs(output_data_path, exist_ok=True)
    zip_path = download_file_from_google_drive(gdrive_file_id, downloaded_zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_data_path)
    return output_data_path


def extract_tracker_data(tracker, conf):
    """
    Get a dictionnary collecting interesting fields of the tracker and experimental settings

    Args:
        tracker (covid19sim.track.Tracker): Tracker toring simulation data
        conf (dict): Experimental Configuration

    returns:
        dict: the extracted data
    """
    timenow = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    data = dict()
    data['intervention_day'] = conf.get('INTERVENTION_DAY')
    data['intervention'] = conf.get('INTERVENTION')
    data['risk_model'] = conf.get('RISK_MODEL')
    data['adoption_rate'] = getattr(tracker, 'adoption_rate', 1.0)
    data['expected_mobility'] = tracker.expected_mobility
    data['serial_interval'] = tracker.get_serial_interval()
    data['all_serial_intervals'] = tracker.serial_intervals
    data['generation_times'] = tracker.get_generation_time()
    data['mobility'] = tracker.mobility
    data['n_init_infected'] = tracker.n_infected_init
    data['contacts'] = dict(tracker.contacts)
    data['cases_per_day'] = tracker.cases_per_day
    data['ei_per_day'] = tracker.ei_per_day
    data['r_0'] = tracker.r_0
    data['R'] = tracker.r
    data['n_humans'] = tracker.n_humans
    data['s'] = tracker.s_per_day
    data['e'] = tracker.e_per_day
    data['i'] = tracker.i_per_day
    data['r'] = tracker.r_per_day
    data['avg_infectiousness_per_day'] = tracker.avg_infectiousness_per_day
    data['risk_precision_global'] = tracker.compute_risk_precision(False)
    data['risk_precision'] = tracker.risk_precision_daily
    data['human_monitor'] = tracker.human_monitor
    data['infection_monitor'] = tracker.infection_monitor
    data['infector_infectee_update_messages'] = tracker.infector_infectee_update_messages
    data['risk_attributes'] = tracker.risk_attributes
    data['feelings'] = tracker.feelings
    data['rec_feelings'] = tracker.rec_feelings
    data['outside_daily_contacts'] = tracker.outside_daily_contacts
    data['test_monitor'] = tracker.test_monitor
    data['encounter_distances'] = tracker.encounter_distances
    data['effective_contacts_since_intervention'] = tracker.compute_effective_contacts(since_intervention=True)
    data['effective_contacts_all_days'] = tracker.compute_effective_contacts(since_intervention=False)
    data['humans_state'] = tracker.humans_state
    data['humans_rec_level'] = tracker.humans_rec_level
    data['humans_intervention_level'] = tracker.humans_intervention_level
    data['humans_has_app'] = dict((human.name, human.has_app) for human in tracker.city.humans)
    data['day_encounters'] = dict(tracker.day_encounters)
    data['daily_age_group_encounters'] = dict(tracker.daily_age_group_encounters)
    data['tracked_humans'] = dict({human.name:human.my_history for human in tracker.city.humans})
    data['age_histogram'] = tracker.city.age_histogram
    data['p_transmission'] = tracker.compute_probability_of_transmission()
    data['covid_properties'] = tracker.covid_properties
    data['human_has_app'] = tracker.human_has_app
    data['to_human_max_msg_per_day'] = tracker.to_human_max_msg_per_day
    # data['dist_encounters'] = dict(tracker.dist_encounters)
    # data['time_encounters'] = dict(tracker.time_encounters)
    # data['day_encounters'] = dict(tracker.day_encounters)
    # data['hour_encounters'] = dict(tracker.hour_encounters)
    # data['daily_age_group_encounters'] = dict(tracker.daily_age_group_encounters)
    # data['age_distribution'] = tracker.age_distribution
    # data['sex_distribution'] = tracker.sex_distribution
    # data['house_size'] = tracker.house_size
    # data['house_age'] = tracker.house_age
    # data['symptoms'] = dict(tracker.symptoms)
    # data['transition_probability'] = dict(tracker.transition_probability)
    return data


def dump_tracker_data(data, outdir, name):
    """
    Writes the tracker's extracted data to outdir/name using dill.

    /!\ there are know incompatibility issues between python 3.7 and 3.8 regarding the dump/loading of data with dill/pickle

    Creates the outputdir if need be, including potential missing parents.

    Args:
        data (dict): tracker's extracted data
        outdir (str): directory where to dump the file
        name (str): the dump file's name
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    with open(outdir / name, 'wb') as f:
        dill.dump(data, f)

def parse_search_configuration(conf):
    """
    Parses the OmegaConf to native types

    Args:
        conf (OmegaConf): Hydra configuration

    Returns:
        dict: parsed conf
    """
    return OmegaConf.to_container(conf, resolve=True)


def parse_configuration(conf):
    """
    Transforms an Omegaconf object to native python dict, parsing specific fields like:
    "1-15" age bin in YAML file becomes (1, 15) tuple, and datetime is parsed from string.

    ANY key-specific parsing should have its inverse in covid19sim.utils.dump_conf()

    Args:
        conf (omegaconf.OmegaConf): Hydra-loaded configuration

    Returns:
        dict: parsed configuration to use in experiment
    """
    if isinstance(conf, (OmegaConf, DictConfig)):
        conf = OmegaConf.to_container(conf, resolve=True)
    elif not isinstance(conf, dict):
        raise ValueError("Unknown configuration type {}".format(type(conf)))

    if "AGE_GROUP_CONTACT_AVG" in conf:
        conf['AGE_GROUP_CONTACT_AVG']['age_groups'] = [
            eval(age_group) for age_group in conf['AGE_GROUP_CONTACT_AVG']['age_groups']
        ]
        conf['AGE_GROUP_CONTACT_AVG']['contact_avg'] = np.array(conf['AGE_GROUP_CONTACT_AVG']['contact_avg'])

    if "SMARTPHONE_OWNER_FRACTION_BY_AGE" in conf:
        conf["SMARTPHONE_OWNER_FRACTION_BY_AGE"] = {
            tuple(int(i) for i in k.split("-")): v
            for k, v in conf["SMARTPHONE_OWNER_FRACTION_BY_AGE"].items()
        }

    if "NORMALIZED_SUSCEPTIBILITY_BY_AGE" in conf:
        conf["NORMALIZED_SUSCEPTIBILITY_BY_AGE"] = {
            tuple(int(i) for i in k.split("-")): v
            for k, v in conf["NORMALIZED_SUSCEPTIBILITY_BY_AGE"].items()
        }

    if "HUMAN_DISTRIBUTION" in conf:
        conf["HUMAN_DISTRIBUTION"] = {
            tuple(int(i) for i in k.split("-")): v
            for k, v in conf["HUMAN_DISTRIBUTION"].items()
        }

    if "MEAN_DAILY_INTERACTION_FOR_AGE_GROUP" in conf:
        conf["MEAN_DAILY_INTERACTION_FOR_AGE_GROUP"] = {
            tuple(int(i) for i in k.split("-")): v
            for k, v in conf["MEAN_DAILY_INTERACTION_FOR_AGE_GROUP"].items()
        }

    if "start_time" in conf:
        conf["start_time"] = datetime.datetime.strptime(
            conf["start_time"], "%Y-%m-%d %H:%M:%S"
        )

    assert "RISK_MODEL" in conf and conf["RISK_MODEL"] is not None

    try:
        conf["GIT_COMMIT_HASH"] = get_git_revision_hash()
    except subprocess.CalledProcessError as e:
        print(">> Contained git error:")
        print(e)
        print(">> Ignoring git hash")
        conf["GIT_COMMIT_HASH"] = "NO_GIT"
    return conf


def dump_conf(
        conf: dict,
        path: typing.Union[str, Path],
):
    """
    Perform a deep copy of the configuration dictionary, preprocess the elements into strings
    to reverse the preprocessing performed by `parse_configuration` and then, dumps the content into a `.yaml` file.

    Args:
        conf (dict): configuration dictionary to be written in a file
        path (str | Path): `.yaml` file where the configuration is written
    """

    copy_conf = deepcopy(conf)

    if "AGE_GROUP_CONTACT_AVG" in copy_conf:
        copy_conf['AGE_GROUP_CONTACT_AVG']['age_groups'] = \
            ["(" + ", ".join([str(i) for i in age_group]) + ")"
             for age_group in copy_conf["AGE_GROUP_CONTACT_AVG"]['age_groups']]
        copy_conf['AGE_GROUP_CONTACT_AVG']['contact_avg'] = copy_conf['AGE_GROUP_CONTACT_AVG']['contact_avg'].tolist()

    if "SMARTPHONE_OWNER_FRACTION_BY_AGE" in copy_conf:
        copy_conf["SMARTPHONE_OWNER_FRACTION_BY_AGE"] = {
            "-".join([str(i) for i in k]): v
            for k, v in copy_conf["SMARTPHONE_OWNER_FRACTION_BY_AGE"].items()
        }

    if "HUMAN_DISTRIBUTION" in copy_conf:
        copy_conf["HUMAN_DISTRIBUTION"] = {
            "-".join([str(i) for i in k]): v
            for k, v in copy_conf["HUMAN_DISTRIBUTION"].items()
        }

    if "NORMALIZED_SUSCEPTIBILITY_BY_AGE" in copy_conf:
        copy_conf["NORMALIZED_SUSCEPTIBILITY_BY_AGE"] = {
                "-".join([str(i) for i in k]): v
                for k, v in copy_conf["NORMALIZED_SUSCEPTIBILITY_BY_AGE"].items()
            }

    if "MEAN_DAILY_INTERACTION_FOR_AGE_GROUP" in copy_conf:
        copy_conf["MEAN_DAILY_INTERACTION_FOR_AGE_GROUP"] = {
                "-".join([str(i) for i in k]): v
                for k, v in copy_conf["MEAN_DAILY_INTERACTION_FOR_AGE_GROUP"].items()
            }

    if "start_time" in copy_conf:
        copy_conf["start_time"] = copy_conf["start_time"].strftime("%Y-%m-%d %H:%M:%S")

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print("WARNING configuration already exists in {}. Overwriting.".format(
            str(path.parent)
        ))

    with path.open("w") as f:
        yaml.safe_dump(copy_conf, f)


def relativefreq2absolutefreq(
        bins_fractions: dict,
        n_elements: int,
        rng
) -> dict:
    """
    Convert relative frequencies to absolute frequencies such that the number of elements sum to n_entity.
    First, we assign `math.floor(fraction*n_entity)` to each bin and then, we assign the remaining elements randomly
    until we have `n_entity`.
    Args:
        bins_fractions (dict): each key is the bin description and each value is the relative frequency.
        n_elements (int): the total number of elements to assign.
        rng: a random generator for randomly assigning the remaining elements
    Returns:
        histogram (dict): each key is the bin description and each value is the absolute frequency.
    """
    histogram = {}
    for my_bin, fraction in bins_fractions.items():
        histogram[my_bin] = math.floor(fraction * n_elements)
    while np.sum(list(histogram.values())) < n_elements:
        bins = list(histogram.keys())
        random_bin = rng.choice(len(bins))
        histogram[bins[random_bin]] += 1

    assert np.sum(list(histogram.values())) == n_elements

    return histogram


def get_git_revision_hash():
    """Get current git hash the code is run from

    Returns:
        str: git hash
    """
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

def get_test_false_negative_rate(test_type, days_since_exposure, conf, interpolate="step"):
    rates = conf['TEST_TYPES'][test_type]['P_FALSE_NEGATIVE']['rate']
    days = conf['TEST_TYPES'][test_type]['P_FALSE_NEGATIVE']['days_since_exposure']
    if interpolate == "step":
        for x, y in zip(days, rates):
            if days_since_exposure <= x:
                return y
        return y
    else:
        raise

def subprocess_cmd(command):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()

def zip_outdir(outdir):
    path = Path(outdir).resolve()
    assert path.exists()
    print(f"Zipping {outdir}...")
    start_time = time.time()
    command = "cd {}; zip -r -0 {}.zip {}".format(
        str(path.parent), path.name, path.name
    )
    subprocess_cmd(command)


def normal_pdf(x, mean, std):
    proba = np.exp(-(((x - mean) ** 2) / (2 * std ** 2))) / (std * (2 * np.pi) ** 0.5)
    return proba