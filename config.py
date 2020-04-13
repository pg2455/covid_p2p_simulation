
# NOISE IN SIM PARAMETERS
LOCATION_TECH = 'gps' # &location-tech

# CITY PARAMETERS
OPENING_HOUR = 9 # @param
CLOSING_HOUR = 18 # @param

# INDIVIDUAL DIFFERENCES PARAMETERS
WORK_FROM_HOME = False
P_HAS_APP = 0.5 # &has_app
P_CAREFUL_PERSON = 0.3 # &carefullness

# DISEASE PARAMETERS
AVG_INCUBATION_DAYS = 5 # &avg-incubation-days
SCALE_INCUBATION_DAYS = 1
AVG_RECOVERY_DAYS = 14
SCALE_RECOVERY_DAYS = 4
INFECTION_RADIUS = 200 # cms
INFECTION_DURATION = 2 # minutes

ASYMPTOMATIC_INFECTION_RATIO = 0.1 # &prob_infectious

RISK_TRANSMISSION_PROBA = 0.01
RISK_WITH_TRUE_SYMPTOMS = False
CLIP_RISK = False

#                   0-9 10-19 20-29  30-39  40-49  50-59 60-69 70-79  80-  # Assuming dath rate to be same for 80 and above
P_NEVER_RECOVERS = [0, 0.002, 0.002, 0.002, 0.004, 0.02, 0.04, 0.08, 0.15] # &never_recovers
REINFECTION_POSSIBLE = 0 # [0, 1]

# aerosol    copper      cardboard       steel       plastic
MAX_DAYS_CONTAMINATION = [0.125, 1/3, 1, 2, 3] # &envrionmental contamination

NUM_DAYS_SICK = 10 # @param
BASELINE_P_ASYMPTOMATIC = 50 # &p-asymptomatic
P_TEST = 0.5
P_FALSE_NEGATIVE = 0.1 #&false-negative # 0  1   2    3   4    5   6    7    8
VIRAL_LOAD_MIN = 0.0001
P_COLD = 0.1 # &p-cold
P_FLU = 0.05 # &p-flu

TEST_DAYS = 2 #

MASK_EFFICACY_NORMIE = 0.32
MASK_EFFICACY_HEALTHWORKER = 0.98

# SIMULATION PARAMETERS
TICK_MINUTE = 2  # @param increment
SIMULATION_DAYS = 30  # @param
SYMPTOM_DAYS = 5  # @param

# LIFESTYLE PARAMETERS
## SHOP
AVG_SHOP_TIME_MINUTES = 30 # @param
SCALE_SHOP_TIME_MINUTES = 15
AVG_SCALE_SHOP_TIME_MINUTES =  10
SCALE_SCALE_SHOP_TIME_MINUTES = 5
NUM_WEEKLY_GROCERY_RUNS = 2 # @param

AVG_MAX_NUM_SHOP_PER_WEEK = 5
SCALE_MAX_NUM_SHOP_PER_WEEK = 1

AVG_NUM_SHOPPING_DAYS = 3
SCALE_NUM_SHOPPING_DAYS = 1
AVG_NUM_SHOPPING_HOURS = 3
SCALE_NUM_SHOPPING_HOURS = 1

## WORK
AVG_WORKING_MINUTES = 8 * 60
SCALE_WORKING_MINUTES = 1 * 60
AVG_SCALE_WORKING_MINUTES = 2 * 60
SCALE_SCALE_WORKING_MINUTES = 1 * 60

## HOSPITAL
AVG_HOSPITAL_HOURS = 7 * 24
SCALE_HOSPITAL_HOURS = 24
AVG_SCALE_HOSPITAL_HOURS = 12
SCALE_SCALE_HOSPITAL_HOURS = 6

## EXERCISE
AVG_EXERCISE_MINUTES = 60
SCALE_EXERCISE_MINUTES = 15
AVG_SCALE_EXERCISE_MINUTES = 15
SCALE_SCALE_EXERCISE_MINUTES = 5

AVG_MAX_NUM_EXERCISE_PER_WEEK = 5
SCALE_MAX_NUM_EXERCISE_PER_WEEK = 2

AVG_NUM_EXERCISE_DAYS = 3
SCALE_NUM_EXERCISE_DAYS = 1
AVG_NUM_EXERCISE_HOURS = 3
SCALE_NUM_EXERCISE_HOURS = 1

## MISC
AVG_MISC_MINUTES = 60
SCALE_MISC_MINUTES = 15
AVG_SCALE_MISC_MINUTES = 15
SCALE_SCALE_MISC_MINUTES = 5

# DISTANCE_ENCOUNTER PARAMETERS
MIN_DIST_ENCOUNTER = 20
MAX_DIST_ENCOUNTER = 50

# VIRAL LOAD PARAMS
MIN_VIRAL_LOAD = 0.1
MAX_VIRAL_LOAD = 0.4
RECOVERY_MEAN = 6
RECOVERY_STD = 5
RECOVERY_CLIP_LOW = 2.5
RECOVERY_CLIP_HIGH = 30
PLATEAU_START_MEAN=2.
PLATEAU_START_STD=8.
PLATEAU_START_CLIP_HIGH = 9.
PLATEAU_START_CLIP_LOW = 0.8
PLATEAU_DURATION_MEAN=5.5
PLEATEAU_DURATION_STD=3.
PLATEAU_DURATION_CLIP_LOW = 3.
PLATEAU_DURATION_CLIP_HIGH = 9.
