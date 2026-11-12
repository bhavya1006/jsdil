from decimal import Decimal

RAW_PERMIT_HISTORY_TRAIN_FILE = "data/permit_history_train.csv"
RAW_PERMIT_HISTORY_EVAL_FILE = "data/permit_history_eval.csv"
RAW_JOB_LOGS_TRAIN_FILE = "data/job_logs_train.csv"
RAW_JOB_LOGS_EVAL_FILE = "data/job_logs_eval.csv"
RAW_STATEMENT_SENTENCES_FILE = "data/statement_sentences.csv"

CLEAN_PERMIT_HISTORY_TRAIN_FILE = "processed/permit_history_train_clean.csv"
CLEAN_PERMIT_HISTORY_EVAL_FILE = "processed/permit_history_eval_clean.csv"
CLEAN_JOB_LOGS_TRAIN_FILE = "processed/job_logs_train_clean.csv"
CLEAN_JOB_LOGS_EVAL_FILE = "processed/job_logs_eval_clean.csv"
CLEAN_STATEMENT_SENTENCES_FILE = "processed/statement_sentences_clean.csv"

DATA_QUALITY_REPORT_FILE = "output/data_quality_report.json"
JSON_INDENT = 2
TARGET_SHARE_DECIMALS = 4
DATASET_NAME_WIDTH = 30
ROW_COUNT_WIDTH = 6


NULL_TOKENS = ["", "n/a", "na", "-", "null", "illegible", "not recorded"]


SITE_PORTAL = "Site portal"
CONTRACTOR_APP = "Contractor app"
SCANNED_PAPER = "Scanned paper"

REQUEST_CHANNEL_MAPPING = {
    "site portal": SITE_PORTAL,
    "contractor app": CONTRACTOR_APP,
    "scanned paper": SCANNED_PAPER,
}


BOOLEAN_YES = "Yes"
BOOLEAN_NO = "No"

BOOLEAN_MAPPING = {
    "yes": BOOLEAN_YES,
    "y": BOOLEAN_YES,
    "true": BOOLEAN_YES,
    "1": BOOLEAN_YES,
    "no": BOOLEAN_NO,
    "n": BOOLEAN_NO,
    "false": BOOLEAN_NO,
    "0": BOOLEAN_NO,
}


HOT_WORK = "Hot work"
CONFINED_SPACE_ENTRY = "Confined space entry"
ELECTRICAL_ISOLATION = "Electrical isolation"
WORK_AT_HEIGHT = "Work at height"
EXCAVATION = "Excavation"

PERMIT_TYPE_MAPPING = {
    "hot work": HOT_WORK,
    "hot work permit": HOT_WORK,
    "hw": HOT_WORK,
    "confined space entry": CONFINED_SPACE_ENTRY,
    "confined space permit": CONFINED_SPACE_ENTRY,
    "cse": CONFINED_SPACE_ENTRY,
    "electrical isolation": ELECTRICAL_ISOLATION,
    "electrical isolation permit": ELECTRICAL_ISOLATION,
    "ei": ELECTRICAL_ISOLATION,
    "work at height": WORK_AT_HEIGHT,
    "work at height permit": WORK_AT_HEIGHT,
    "wah": WORK_AT_HEIGHT,
    "excavation": EXCAVATION,
    "excavation permit": EXCAVATION,
    "exc": EXCAVATION,
}

NO_LINKED_PERMIT = "No linked permit"

LINKED_PERMIT_TYPE_MAPPING = {
    **PERMIT_TYPE_MAPPING,
    "no linked permit": NO_LINKED_PERMIT,
    "no link": NO_LINKED_PERMIT,
    "nil": NO_LINKED_PERMIT,
}


ZONE_1 = "Zone 1"
ZONE_2 = "Zone 2"
NON_HAZARDOUS = "Non-hazardous"

AREA_CLASSIFICATION_MAPPING = {
    "zone 1": ZONE_1,
    "z1": ZONE_1,
    "zone-1": ZONE_1,
    "zone 2": ZONE_2,
    "z2": ZONE_2,
    "zone-2": ZONE_2,
    "non-hazardous": NON_HAZARDOUS,
    "safe area": NON_HAZARDOUS,
    "non-haz": NON_HAZARDOUS,
}


OPERATING = "Operating"
SHUT_DOWN = "Shut down"
NO_PROCESS_PLANT = "No process plant"

PROCESS_UNIT_STATUS_MAPPING = {
    "operating": OPERATING,
    "running": OPERATING,
    "oper": OPERATING,
    "shut down": SHUT_DOWN,
    "off-line": SHUT_DOWN,
    "s/d": SHUT_DOWN,
    "no process plant": NO_PROCESS_PLANT,
    "no plant": NO_PROCESS_PLANT,
}


INDOORS = "Indoors"
OUTDOORS = "Outdoors"

WORK_LOCATION_MAPPING = {
    "indoors": INDOORS,
    "inside": INDOORS,
    "in": INDOORS,
    "outdoors": OUTDOORS,
    "outside": OUTDOORS,
    "out": OUTDOORS,
}


SCAFFOLD = "Scaffold"
MOBILE_ELEVATING_WORK_PLATFORM = "Mobile elevating work platform"
LADDER = "Ladder"
FIXED_PLATFORM = "Fixed platform"
ROPE_ACCESS = "Rope access"

ACCESS_METHOD_MAPPING = {
    "scaffold": SCAFFOLD,
    "scaf": SCAFFOLD,
    "mobile elevating work platform": MOBILE_ELEVATING_WORK_PLATFORM,
    "mewp": MOBILE_ELEVATING_WORK_PLATFORM,
    "ladder": LADDER,
    "fixed platform": FIXED_PLATFORM,
    "platform": FIXED_PLATFORM,
    "rope access": ROPE_ACCESS,
    "rope": ROPE_ACCESS,
}


REACTOR_HOUSE_A = "Reactor House A"
REACTOR_HOUSE_B = "Reactor House B"
DISTILLATION_UNIT = "Distillation Unit"
SOLVENT_RECOVERY = "Solvent Recovery"
CHLORINE_UNIT = "Chlorine Unit"
CAUSTIC_EVAPORATION = "Caustic Evaporation"
DRUM_FILLING = "Drum Filling"
FLARE_AREA = "Flare Area"
EAST_PIPE_BRIDGE = "East Pipe Bridge"
WEST_PIPE_BRIDGE = "West Pipe Bridge"
TANK_FARM_NORTH = "Tank Farm North"
TANK_FARM_SOUTH = "Tank Farm South"
ROAD_LOADING_BAY = "Road Loading Bay"
RAIL_SIDINGS = "Rail Sidings"
EFFLUENT_TREATMENT = "Effluent Treatment"
BOILER_HOUSE = "Boiler House"
COOLING_TOWERS = "Cooling Towers"
COMPRESSOR_HOUSE = "Compressor House"
MAIN_SUBSTATION = "Main Substation"
WATER_TREATMENT = "Water Treatment"
SITE_ROADS_AND_DRAINS = "Site Roads and Drains"
ENGINEERING_WORKSHOPS = "Engineering Workshops"
CENTRAL_WAREHOUSE = "Central Warehouse"
LABORATORY = "Laboratory"
CONTROL_BUILDING = "Control Building"
ADMINISTRATION_BLOCK = "Administration Block"
GATEHOUSE = "Gatehouse"
FIRE_STATION = "Fire Station"
WEIGHBRIDGE = "Weighbridge"

PLANT_AREA_MAPPING = {
    "reactor house a": REACTOR_HOUSE_A,
    "reactor house b": REACTOR_HOUSE_B,
    "distillation unit": DISTILLATION_UNIT,
    "solvent recovery": SOLVENT_RECOVERY,
    "chlorine unit": CHLORINE_UNIT,
    "caustic evaporation": CAUSTIC_EVAPORATION,
    "drum filling": DRUM_FILLING,
    "flare area": FLARE_AREA,
    "east pipe bridge": EAST_PIPE_BRIDGE,
    "west pipe bridge": WEST_PIPE_BRIDGE,
    "tank farm north": TANK_FARM_NORTH,
    "tank farm south": TANK_FARM_SOUTH,
    "road loading bay": ROAD_LOADING_BAY,
    "rail sidings": RAIL_SIDINGS,
    "effluent treatment": EFFLUENT_TREATMENT,
    "boiler house": BOILER_HOUSE,
    "cooling towers": COOLING_TOWERS,
    "compressor house": COMPRESSOR_HOUSE,
    "main substation": MAIN_SUBSTATION,
    "water treatment": WATER_TREATMENT,
    "site roads and drains": SITE_ROADS_AND_DRAINS,
    "engineering workshops": ENGINEERING_WORKSHOPS,
    "central warehouse": CENTRAL_WAREHOUSE,
    "laboratory": LABORATORY,
    "control building": CONTROL_BUILDING,
    "administration block": ADMINISTRATION_BLOCK,
    "gatehouse": GATEHOUSE,
    "fire station": FIRE_STATION,
    "weighbridge": WEIGHBRIDGE,
}


WELDING = "Welding"
GRINDING = "Grinding"
FLAME_CUTTING = "Flame cutting"
BRAZING = "Brazing"
HOT_TAPPING = "Hot tapping"
VESSEL_INSPECTION = "Vessel inspection"
TANK_CLEANING = "Tank cleaning"
INTERNAL_REPAIR = "Internal repair"
PIT_AND_SUMP_ENTRY = "Pit and sump entry"
MOTOR_MAINTENANCE = "Motor maintenance"
SWITCHGEAR_MAINTENANCE = "Switchgear maintenance"
CABLE_TERMINATION = "Cable termination"
LIGHTING_AND_SMALL_POWER = "Lighting and small power"
HEATER_ELEMENT_REPLACEMENT = "Heater element replacement"
PIPEWORK_AT_HEIGHT = "Pipework at height"
INSULATION_AND_CLADDING = "Insulation and cladding"
TANK_ROOF_WORK = "Tank roof work"
ROOF_REPAIR = "Roof repair"
PAINTING = "Painting"
CABLE_TRAY_WORK = "Cable tray work"
TRIAL_HOLE = "Trial hole"
CABLE_TRENCHING = "Cable trenching"
DRAIN_REPAIR = "Drain repair"
FOUNDATION_WORK = "Foundation work"
BURIED_PIPE_REPAIR = "Buried pipe repair"

WORK_ACTIVITY_MAPPING = {
    "welding": WELDING,
    "grinding": GRINDING,
    "flame cutting": FLAME_CUTTING,
    "brazing": BRAZING,
    "hot tapping": HOT_TAPPING,
    "vessel inspection": VESSEL_INSPECTION,
    "tank cleaning": TANK_CLEANING,
    "internal repair": INTERNAL_REPAIR,
    "pit and sump entry": PIT_AND_SUMP_ENTRY,
    "motor maintenance": MOTOR_MAINTENANCE,
    "switchgear maintenance": SWITCHGEAR_MAINTENANCE,
    "cable termination": CABLE_TERMINATION,
    "lighting and small power": LIGHTING_AND_SMALL_POWER,
    "heater element replacement": HEATER_ELEMENT_REPLACEMENT,
    "pipework at height": PIPEWORK_AT_HEIGHT,
    "insulation and cladding": INSULATION_AND_CLADDING,
    "tank roof work": TANK_ROOF_WORK,
    "roof repair": ROOF_REPAIR,
    "painting": PAINTING,
    "cable tray work": CABLE_TRAY_WORK,
    "trial hole": TRIAL_HOLE,
    "cable trenching": CABLE_TRENCHING,
    "drain repair": DRAIN_REPAIR,
    "foundation work": FOUNDATION_WORK,
    "buried pipe repair": BURIED_PIPE_REPAIR,
}


OUTCOME_ISSUED_AS_REQUESTED = "Issued as requested"
OUTCOME_ISSUED_AFTER_REWORK = "Issued after rework"
OUTCOME_ISSUED_AFTER_SIMOPS_REVIEW = "Issued after SIMOPS review"

REVIEW_OUTCOME_MAPPING = {
    "issued as requested": OUTCOME_ISSUED_AS_REQUESTED,
    "issued": OUTCOME_ISSUED_AS_REQUESTED,
    "iss": OUTCOME_ISSUED_AS_REQUESTED,
    "issued after rework": OUTCOME_ISSUED_AFTER_REWORK,
    "issued - rework": OUTCOME_ISSUED_AFTER_REWORK,
    "iss-rw": OUTCOME_ISSUED_AFTER_REWORK,
    "issued after simops review": OUTCOME_ISSUED_AFTER_SIMOPS_REVIEW,
    "issued - simops": OUTCOME_ISSUED_AFTER_SIMOPS_REVIEW,
    "iss-sim": OUTCOME_ISSUED_AFTER_SIMOPS_REVIEW,
}


GAS_ALARM = "Gas alarm"
PROCESS_UPSET = "Process upset"
SITE_ALARM = "Site alarm"
CONFLICTING_WORK = "Conflicting work"
UNSAFE_CONDITION = "Unsafe condition"
HIGH_WIND = "High wind"

SUSPENSION_REASON_MAPPING = {
    "gas alarm": GAS_ALARM,
    "gas": GAS_ALARM,
    "process upset": PROCESS_UPSET,
    "proc": PROCESS_UPSET,
    "site alarm": SITE_ALARM,
    "alarm": SITE_ALARM,
    "conflicting work": CONFLICTING_WORK,
    "conflict": CONFLICTING_WORK,
    "simops": CONFLICTING_WORK,
    "unsafe condition": UNSAFE_CONDITION,
    "unsafe": UNSAFE_CONDITION,
    "high wind": HIGH_WIND,
    "wind": HIGH_WIND,
}


COMPLETED = "Completed"
COMPLETED_AFTER_EXTENSION = "Completed after extension"
COMPLETED_AFTER_SUSPENSION = "Completed after suspension"

COMPLETION_CODE_MAPPING = {
    "completed": COMPLETED,
    "done": COMPLETED,
    "comp": COMPLETED,
    "completed after extension": COMPLETED_AFTER_EXTENSION,
    "done - extended": COMPLETED_AFTER_EXTENSION,
    "comp-ext": COMPLETED_AFTER_EXTENSION,
    "completed after suspension": COMPLETED_AFTER_SUSPENSION,
    "done - suspended": COMPLETED_AFTER_SUSPENSION,
    "comp-sus": COMPLETED_AFTER_SUSPENSION,
}


METHOD_STATEMENT = "Method statement"
TASK_RISK_ASSESSMENT = "Task risk assessment"
GAS_TEST_RECORD = "Gas test record"
RESCUE_PLAN = "Rescue plan"

DOCUMENT_TYPE_MAPPING = {
    "method statement": METHOD_STATEMENT,
    "task risk assessment": TASK_RISK_ASSESSMENT,
    "gas test record": GAS_TEST_RECORD,
    "rescue plan": RESCUE_PLAN,
}


ISOLATION = "Isolation"
GAS_TESTING = "Gas testing"
FIRE_PRECAUTIONS = "Fire precautions"
RESCUE = "Rescue"
ACCESS = "Access"
PERSONAL_PROTECTIVE_EQUIPMENT = "Personal protective equipment"
SIMULTANEOUS_OPERATIONS = "Simultaneous operations"
HOUSEKEEPING_AND_WASTE = "Housekeeping and waste"
BACKGROUND = "Background"

SENTENCE_TYPE_MAPPING = {
    "isolation": ISOLATION,
    "isolations": ISOLATION,
    "gas testing": GAS_TESTING,
    "gas test": GAS_TESTING,
    "gas-testing": GAS_TESTING,
    "fire precautions": FIRE_PRECAUTIONS,
    "fire": FIRE_PRECAUTIONS,
    "fire-precautions": FIRE_PRECAUTIONS,
    "rescue": RESCUE,
    "emergency rescue": RESCUE,
    "access": ACCESS,
    "access and egress": ACCESS,
    "personal protective equipment": PERSONAL_PROTECTIVE_EQUIPMENT,
    "ppe": PERSONAL_PROTECTIVE_EQUIPMENT,
    "personal-protective-equipment": PERSONAL_PROTECTIVE_EQUIPMENT,
    "simultaneous operations": SIMULTANEOUS_OPERATIONS,
    "simops": SIMULTANEOUS_OPERATIONS,
    "simultaneous-operations": SIMULTANEOUS_OPERATIONS,
    "housekeeping and waste": HOUSEKEEPING_AND_WASTE,
    "housekeeping & waste": HOUSEKEEPING_AND_WASTE,
    "housekeeping": HOUSEKEEPING_AND_WASTE,
    "housekeeping-and-waste": HOUSEKEEPING_AND_WASTE,
    "background": BACKGROUND,
    "context": BACKGROUND,
    "general": BACKGROUND,
}


PERMIT_REFERENCE_PATTERN = r"^(?:PTW(?=[0-9]{7}$)|PTW-)?([0-9]{1,7})$"
PERMIT_REFERENCE_PREFIX = "PTW"
PERMIT_REFERENCE_DIGITS = 7
CANONICAL_PERMIT_REFERENCE_PATTERN = r"^PTW[0-9]{7}$"
CONTRACTOR_REFERENCE_PATTERN = r"^CTR-[0-9]{4}$"
SENTENCE_ID_PATTERN = r"^SEN-[0-9]{7}$"

WHOLE_NUMBER_PATTERN = r"^([0-9]+)$"
POSITIVE_WHOLE_NUMBER_PATTERN = r"^([1-9][0-9]*)$"
WHOLE_NUMBER_DTYPE = "Int64"
DECIMAL_DTYPE = "float64"
TEXT_DTYPE = "str"

WHOLE_STEP = Decimal("1")
TENTH_STEP = Decimal("0.1")
NO_CONVERSION = Decimal("1")
METRES_PER_FOOT = Decimal("0.3048")
METRES_PER_SECOND_PER_MILE_PER_HOUR = Decimal("0.44704")
HYDROGEN_SULPHIDE_MILLIGRAMS_PER_PPM = Decimal("1.42")
CARBON_MONOXIDE_MILLIGRAMS_PER_PPM = Decimal("1.17")
VOLTS_PER_KILOVOLT = Decimal("1000")
MINUTES_PER_HOUR = Decimal("60")

FLAMMABLE_GAS_READING = {
    "notations": {"percent_lel": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?)(?:%LEL| % LEL)?$"},
    "multipliers": {"percent_lel": NO_CONVERSION},
    "divisors": {"percent_lel": NO_CONVERSION},
    "step": WHOLE_STEP,
    "dtype": WHOLE_NUMBER_DTYPE,
}

OXYGEN_READING = {
    "notations": {"percent": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?)(?:%| % O2)?$"},
    "multipliers": {"percent": NO_CONVERSION},
    "divisors": {"percent": NO_CONVERSION},
    "step": TENTH_STEP,
    "dtype": DECIMAL_DTYPE,
}

HYDROGEN_SULPHIDE_READING = {
    "notations": {"ppm": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?)(?: ppm)?$",
                  "milligrams_per_cubic_metre": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?) mg/m3$"},
    "multipliers": {"ppm": NO_CONVERSION, "milligrams_per_cubic_metre": NO_CONVERSION},
    "divisors": {"ppm": NO_CONVERSION,
                 "milligrams_per_cubic_metre": HYDROGEN_SULPHIDE_MILLIGRAMS_PER_PPM},
    "step": TENTH_STEP,
    "dtype": DECIMAL_DTYPE,
}

CARBON_MONOXIDE_READING = {
    "notations": {"ppm": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?)(?: ppm)?$",
                  "milligrams_per_cubic_metre": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?) mg/m3$"},
    "multipliers": {"ppm": NO_CONVERSION, "milligrams_per_cubic_metre": NO_CONVERSION},
    "divisors": {"ppm": NO_CONVERSION,
                 "milligrams_per_cubic_metre": CARBON_MONOXIDE_MILLIGRAMS_PER_PPM},
    "step": TENTH_STEP,
    "dtype": DECIMAL_DTYPE,
}

LENGTH_READING = {
    "notations": {"metres": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?)(?: m)?$",
                  "feet": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?) ft$"},
    "multipliers": {"metres": NO_CONVERSION, "feet": METRES_PER_FOOT},
    "divisors": {"metres": NO_CONVERSION, "feet": NO_CONVERSION},
    "step": TENTH_STEP,
    "dtype": DECIMAL_DTYPE,
}

WIND_SPEED_READING = {
    "notations": {"metres_per_second": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?)(?: m/s)?$",
                  "miles_per_hour": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?) mph$"},
    "multipliers": {"metres_per_second": NO_CONVERSION,
                    "miles_per_hour": METRES_PER_SECOND_PER_MILE_PER_HOUR},
    "divisors": {"metres_per_second": NO_CONVERSION, "miles_per_hour": NO_CONVERSION},
    "step": TENTH_STEP,
    "dtype": DECIMAL_DTYPE,
}

SYSTEM_VOLTAGE_READING = {
    "notations": {"volts": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?)V?$",
                  "kilovolts": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?) ?kV$"},
    "multipliers": {"volts": NO_CONVERSION, "kilovolts": VOLTS_PER_KILOVOLT},
    "divisors": {"volts": NO_CONVERSION, "kilovolts": NO_CONVERSION},
    "step": WHOLE_STEP,
    "dtype": WHOLE_NUMBER_DTYPE,
}

FIRE_WATCH_READING = {
    "notations": {"minutes": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?)(?: min)?$",
                  "hours": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?) hrs?$"},
    "multipliers": {"minutes": NO_CONVERSION, "hours": MINUTES_PER_HOUR},
    "divisors": {"minutes": NO_CONVERSION, "hours": NO_CONVERSION},
    "step": WHOLE_STEP,
    "dtype": WHOLE_NUMBER_DTYPE,
}

WORK_HOURS_READING = {
    "notations": {"hours": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?)(?: hrs?)?$",
                  "minutes": r"(?i)^(-?[0-9]+(?:\.[0-9]+)?) min$"},
    "multipliers": {"hours": NO_CONVERSION, "minutes": NO_CONVERSION},
    "divisors": {"hours": NO_CONVERSION, "minutes": MINUTES_PER_HOUR},
    "step": TENTH_STEP,
    "dtype": DECIMAL_DTYPE,
}

GAS_READING_MINIMUM = 0
GAS_PERCENT_MAXIMUM = 100
LENGTH_MINIMUM = float(TENTH_STEP)
WIND_SPEED_MINIMUM = 0
FIRE_WATCH_MINIMUM = 1
SYSTEM_VOLTAGE_MINIMUM = 1
WORK_HOURS_MINIMUM = float(TENTH_STEP)
WORK_PARTY_MINIMUM = 1


MARKUP_PATTERN = r"</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>"
ENCODING_ARTIFACT_PATTERN = r"&[#a-z0-9]+;|\xa0"
WHITESPACE_RUN_PATTERN = r"[\s\xa0]+"
MINIMUM_SENTENCE_WORDS = 3


PERMIT_REFERENCE_COLUMN = "permit_reference"
REVISION_NUMBER_COLUMN = "revision_number"

FORM_VALUE_MAPPINGS = {
    "request_channel": REQUEST_CHANNEL_MAPPING,
    "plant_area": PLANT_AREA_MAPPING,
    "area_classification": AREA_CLASSIFICATION_MAPPING,
    "process_unit_status": PROCESS_UNIT_STATUS_MAPPING,
    "work_location": WORK_LOCATION_MAPPING,
    "permit_type": PERMIT_TYPE_MAPPING,
    "linked_permit_type": LINKED_PERMIT_TYPE_MAPPING,
    "work_activity": WORK_ACTIVITY_MAPPING,
    "risk_assessment_attached": BOOLEAN_MAPPING,
    "rescue_plan_in_place": BOOLEAN_MAPPING,
    "isolation_proved_dead": BOOLEAN_MAPPING,
    "personal_locks_applied": BOOLEAN_MAPPING,
    "circuit_earths_applied": BOOLEAN_MAPPING,
    "access_method": ACCESS_METHOD_MAPPING,
    "fall_protection_fitted": BOOLEAN_MAPPING,
    "scaffold_inspected": BOOLEAN_MAPPING,
    "services_located": BOOLEAN_MAPPING,
    "trench_support_fitted": BOOLEAN_MAPPING,
    "work_area_barriered": BOOLEAN_MAPPING,
}

FORM_IDENTIFIER_PATTERNS = {
    "contractor_reference": CONTRACTOR_REFERENCE_PATTERN,
}

FORM_WHOLE_NUMBER_PATTERNS = {
    "revision_number": POSITIVE_WHOLE_NUMBER_PATTERN,
    "work_party_size": WHOLE_NUMBER_PATTERN,
}

FORM_READINGS = {
    "planned_work_hours": WORK_HOURS_READING,
    "flammable_gas_lel_percent": FLAMMABLE_GAS_READING,
    "oxygen_percent": OXYGEN_READING,
    "hydrogen_sulphide_ppm": HYDROGEN_SULPHIDE_READING,
    "carbon_monoxide_ppm": CARBON_MONOXIDE_READING,
    "combustibles_cleared_radius_m": LENGTH_READING,
    "fire_watch_minutes": FIRE_WATCH_READING,
    "system_voltage_volts": SYSTEM_VOLTAGE_READING,
    "working_height_m": LENGTH_READING,
    "wind_speed_mps": WIND_SPEED_READING,
    "excavation_depth_m": LENGTH_READING,
}

FORM_MINIMUM_VALUES = {
    "work_party_size": WORK_PARTY_MINIMUM,
    "planned_work_hours": WORK_HOURS_MINIMUM,
    "flammable_gas_lel_percent": GAS_READING_MINIMUM,
    "oxygen_percent": GAS_READING_MINIMUM,
    "hydrogen_sulphide_ppm": GAS_READING_MINIMUM,
    "carbon_monoxide_ppm": GAS_READING_MINIMUM,
    "combustibles_cleared_radius_m": LENGTH_MINIMUM,
    "fire_watch_minutes": FIRE_WATCH_MINIMUM,
    "system_voltage_volts": SYSTEM_VOLTAGE_MINIMUM,
    "working_height_m": LENGTH_MINIMUM,
    "wind_speed_mps": WIND_SPEED_MINIMUM,
    "excavation_depth_m": LENGTH_MINIMUM,
}

FORM_MAXIMUM_VALUES = {
    "flammable_gas_lel_percent": GAS_PERCENT_MAXIMUM,
    "oxygen_percent": GAS_PERCENT_MAXIMUM,
}


HISTORY_TARGET_COLUMN = "permit_suspended"

HISTORY_VALUE_MAPPINGS = {
    **FORM_VALUE_MAPPINGS,
    "review_outcome": REVIEW_OUTCOME_MAPPING,
    "permit_suspended": BOOLEAN_MAPPING,
    "suspension_reason": SUSPENSION_REASON_MAPPING,
}

HISTORY_WHOLE_NUMBER_PATTERNS = {
    **FORM_WHOLE_NUMBER_PATTERNS,
    "concurrent_live_permits": WHOLE_NUMBER_PATTERN,
    "conflicting_live_permits": WHOLE_NUMBER_PATTERN,
}


LOG_TARGET_COLUMN = "actual_work_hours"

LOG_VALUE_MAPPINGS = {
    "request_channel": REQUEST_CHANNEL_MAPPING,
    "plant_area": PLANT_AREA_MAPPING,
    "area_classification": AREA_CLASSIFICATION_MAPPING,
    "process_unit_status": PROCESS_UNIT_STATUS_MAPPING,
    "work_location": WORK_LOCATION_MAPPING,
    "permit_type": PERMIT_TYPE_MAPPING,
    "linked_permit_type": LINKED_PERMIT_TYPE_MAPPING,
    "work_activity": WORK_ACTIVITY_MAPPING,
    "access_method": ACCESS_METHOD_MAPPING,
    "trench_support_fitted": BOOLEAN_MAPPING,
    "permit_suspended": BOOLEAN_MAPPING,
    "completion_code": COMPLETION_CODE_MAPPING,
}

LOG_WHOLE_NUMBER_PATTERNS = {
    "work_party_size": WHOLE_NUMBER_PATTERN,
    "concurrent_live_permits": WHOLE_NUMBER_PATTERN,
    "conflicting_live_permits": WHOLE_NUMBER_PATTERN,
}

LOG_READINGS = {
    "fire_watch_minutes": FIRE_WATCH_READING,
    "system_voltage_volts": SYSTEM_VOLTAGE_READING,
    "working_height_m": LENGTH_READING,
    "excavation_depth_m": LENGTH_READING,
    "wind_speed_mps": WIND_SPEED_READING,
    "planned_work_hours": WORK_HOURS_READING,
    "actual_work_hours": WORK_HOURS_READING,
}

LOG_MINIMUM_VALUES = {
    "work_party_size": WORK_PARTY_MINIMUM,
    "fire_watch_minutes": FIRE_WATCH_MINIMUM,
    "system_voltage_volts": SYSTEM_VOLTAGE_MINIMUM,
    "working_height_m": LENGTH_MINIMUM,
    "excavation_depth_m": LENGTH_MINIMUM,
    "wind_speed_mps": WIND_SPEED_MINIMUM,
    "planned_work_hours": WORK_HOURS_MINIMUM,
    "actual_work_hours": WORK_HOURS_MINIMUM,
}


SENTENCE_TARGET_COLUMN = "sentence_type"
SENTENCE_ID_COLUMN = "sentence_id"
SENTENCE_TEXT_COLUMN = "sentence_text"
SENTENCE_LENGTH_COLUMN = "sentence_length_characters"

SENTENCE_VALUE_MAPPINGS = {
    "request_channel": REQUEST_CHANNEL_MAPPING,
    "document_type": DOCUMENT_TYPE_MAPPING,
    "sentence_type": SENTENCE_TYPE_MAPPING,
}

SENTENCE_IDENTIFIER_PATTERNS = {
    SENTENCE_ID_COLUMN: SENTENCE_ID_PATTERN,
}

SENTENCE_WHOLE_NUMBER_PATTERNS = {
    "sentence_number": POSITIVE_WHOLE_NUMBER_PATTERN,
}


DOCS_DIRECTORY = "data/docs"
PDF_FILE_SUFFIX = ".pdf"
DOCUMENT_HEADER_PATTERN = (r"Document\s+[A-Za-z0-9 ]+\|\s*Issue\s*[0-9]+.*?Review cycle:\s*every\s+"
                           r"[a-z]+\s+months")
SECTION_HEADING_PATTERN = r"(?m)^([0-9]{1,2})\. [A-Z]"
SENTENCE_BOUNDARY_PATTERN = r"(?<=[.:;])(?<![0-9]\.)\s+"
CHUNK_SIZE_CHARACTERS = 700
CHUNK_OVERLAP_CHARACTERS = 120

DATABASE_DIRECTORY = "database"
VECTOR_STORE_DIRECTORY = f"{DATABASE_DIRECTORY}/chroma_store"
VECTOR_STORE_FILE_NAME = "chroma.sqlite3"
VECTOR_COLLECTION_NAME = "permit_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
VECTOR_DISTANCE_METRIC = "cosine"
NO_SECTION_PASSAGES_MESSAGE = ("The knowledge index holds no passage of section {section_number} "
                               "of {document_name}")


PERMIT_STANDARD_DOCUMENTS = {
    HOT_WORK: "Permit-Standard-Hot-Work.pdf",
    CONFINED_SPACE_ENTRY: "Permit-Standard-Confined-Space-Entry.pdf",
    ELECTRICAL_ISOLATION: "Permit-Standard-Electrical-Isolation.pdf",
    WORK_AT_HEIGHT: "Permit-Standard-Work-At-Height.pdf",
    EXCAVATION: "Permit-Standard-Excavation.pdf",
}
NO_STANDARD_CONFIGURED_MESSAGE = "No permit standard is configured for {permit_type}"
PERMIT_TYPE_SECTION_NUMBER = 2
REQUIREMENTS_SECTION_NUMBER = 4
DOCUMENTATION_SECTION_NUMBER = 6
PERMIT_TYPE_PATTERN = r"The permit type is ([A-Z][a-z]+(?: [a-z]+)*)\."
REQUIREMENTS_SECTION_PATTERN = (r"4\. Requirements .+? in this order: (.+) A requirement is never "
                                r"met by a sentence in a statement, and is never waived\.")
REQUIREMENT_NUMBER_PATTERN = r"(?:^|(?<=\.) )([1-9]) (?=[A-Z])"
REQUIREMENT_NAME_PART = "requirement_name"
REQUIREMENT_SENTENCE_PART = "requirement_sentence"
REQUIREMENT_NAME_PATTERN = rf"(?P<{REQUIREMENT_NAME_PART}>[A-Z][a-z-]*(?: [a-z-]+)*)\. "
FIGURE_PATTERN = r"[0-9]+(?:\.[0-9]+)?"

RISK_ASSESSMENT_REQUIREMENT = "risk_assessment"
FLAMMABLE_GAS_REQUIREMENT = "flammable_gas"
OXYGEN_REQUIREMENT = "oxygen"
COMBUSTIBLES_CLEARED_REQUIREMENT = "combustibles_cleared"
FIRE_WATCH_REQUIREMENT = "fire_watch"
HYDROGEN_SULPHIDE_REQUIREMENT = "hydrogen_sulphide"
CARBON_MONOXIDE_REQUIREMENT = "carbon_monoxide"
RESCUE_PLAN_REQUIREMENT = "rescue_plan"
PROVED_DEAD_REQUIREMENT = "proved_dead"
PERSONAL_LOCKS_REQUIREMENT = "personal_locks"
CIRCUIT_EARTHS_REQUIREMENT = "circuit_earths"
FALL_PROTECTION_REQUIREMENT = "fall_protection"
SCAFFOLD_INSPECTION_REQUIREMENT = "scaffold_inspection"
BARRIERS_REQUIREMENT = "barriers"
WIND_SPEED_REQUIREMENT = "wind_speed"
SERVICES_LOCATED_REQUIREMENT = "services_located"
TRENCH_SUPPORT_REQUIREMENT = "trench_support"

MINIMUM_RULE = "minimum"
MAXIMUM_RULE = "maximum"
THRESHOLD_RULE = "threshold"

REQUIREMENT_SENTENCE_PATTERNS = {
    RISK_ASSESSMENT_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"A task risk assessment for the job is attached to the request\.)"),
    FLAMMABLE_GAS_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"The flammable gas reading at the work point is at most "
        rf"(?P<{MAXIMUM_RULE}>{FIGURE_PATTERN}) percent of the lower explosive limit\.)"),
    OXYGEN_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        rf"The oxygen reading is at least (?P<{MINIMUM_RULE}>{FIGURE_PATTERN}) percent and at most "
        rf"(?P<{MAXIMUM_RULE}>{FIGURE_PATTERN}) percent\.)"),
    COMBUSTIBLES_CLEARED_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"Combustible materials are cleared to at least "
        rf"(?P<{MINIMUM_RULE}>{FIGURE_PATTERN}) metres around the work\.)"),
    FIRE_WATCH_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        rf"A fire watch stays for at least (?P<{MINIMUM_RULE}>{FIGURE_PATTERN}) minutes after the "
        r"work ends\.)"),
    HYDROGEN_SULPHIDE_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        rf"The hydrogen sulphide reading is at most (?P<{MAXIMUM_RULE}>{FIGURE_PATTERN}) ppm\.)"),
    CARBON_MONOXIDE_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        rf"The carbon monoxide reading is at most (?P<{MAXIMUM_RULE}>{FIGURE_PATTERN}) ppm\.)"),
    RESCUE_PLAN_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"A rescue plan is in place with a standby person at the entry\.)"),
    PROVED_DEAD_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"The isolation is proved dead at the point of work\.)"),
    PERSONAL_LOCKS_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"Every member of the work party has applied a personal lock to the isolation\.)"),
    CIRCUIT_EARTHS_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        rf"On a system above (?P<{THRESHOLD_RULE}>{FIGURE_PATTERN}) volts, circuit earths are "
        r"applied at the point of work\.)"),
    FALL_PROTECTION_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"Fall protection is fitted, as edge protection or a harness clipped to an anchor\.)"),
    SCAFFOLD_INSPECTION_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"A scaffold used for access is inspected and tagged\.)"),
    BARRIERS_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"The area below or around the work is barriered\.)"),
    WIND_SPEED_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        rf"Outdoors, the wind at the work position is at most (?P<{MAXIMUM_RULE}>{FIGURE_PATTERN}) "
        r"metres per second\.)"),
    SERVICES_LOCATED_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        r"Underground services are located before digging starts\.)"),
    TRENCH_SUPPORT_REQUIREMENT: (
        REQUIREMENT_NAME_PATTERN + rf"(?P<{REQUIREMENT_SENTENCE_PART}>"
        rf"A trench or pit deeper than (?P<{THRESHOLD_RULE}>{FIGURE_PATTERN}) metres has its sides "
        r"supported\.)"),
}


REQUIREMENT_MET = "Met"
REQUIREMENT_NOT_MET = "Not met"
REQUIREMENT_NOT_DOCUMENTED = "Not documented"

DECISION_ISSUE = "ISSUE PERMIT"
DECISION_RETURN = "RETURN TO REQUESTER"
DECISION_REFER = "REFER FOR SIMOPS REVIEW"
DECISIONS_OPEN = {
    DECISION_ISSUE: [DECISION_ISSUE, DECISION_RETURN, DECISION_REFER],
    DECISION_RETURN: [DECISION_RETURN, DECISION_REFER],
    DECISION_REFER: [DECISION_RETURN, DECISION_REFER],
}

CONFLICTING_PERMIT_TRIGGER = "conflicting live permit"
LIVE_PROCESS_TRIGGER = "live process unit"
HIGH_VOLTAGE_TRIGGER = "high voltage"
LIVE_PROCESS_PERMIT_TYPES = [HOT_WORK, CONFINED_SPACE_ENTRY]
NOT_LIVE_UNIT_STATUSES = [SHUT_DOWN, NO_PROCESS_PLANT]
HIGH_VOLTAGE_PERMIT_TYPES = [ELECTRICAL_ISOLATION]
HIGH_VOLTAGE_ABOVE_VOLTS = Decimal("1000")
NO_STANDARD_READ_MESSAGE = ("Request {permit_reference} cannot be screened: no permit standard has "
                            "been read for {permit_type}")
STANDARD_OF_ANOTHER_PERMIT_TYPE_MESSAGE = ("The standard read for {permit_type} is "
                                           "{document_name}, which states the permit type "
                                           "{stated_permit_type}")

PERMIT_TYPE_COLUMN = "permit_type"
LINKED_PERMIT_TYPE_COLUMN = "linked_permit_type"
PROCESS_UNIT_STATUS_COLUMN = "process_unit_status"
WORK_LOCATION_COLUMN = "work_location"
ACCESS_METHOD_COLUMN = "access_method"
SYSTEM_VOLTAGE_COLUMN = "system_voltage_volts"
EXCAVATION_DEPTH_COLUMN = "excavation_depth_m"
WIND_SPEED_COLUMN = "wind_speed_mps"
OXYGEN_COLUMN = "oxygen_percent"
CIRCUIT_EARTHS_COLUMN = "circuit_earths_applied"
SCAFFOLD_INSPECTED_COLUMN = "scaffold_inspected"
TRENCH_SUPPORT_COLUMN = "trench_support_fitted"
CONCURRENT_LIVE_PERMITS_COLUMN = "concurrent_live_permits"
CONFLICTING_LIVE_PERMITS_COLUMN = "conflicting_live_permits"

REQUIREMENT_ANSWER_COLUMNS = {
    RISK_ASSESSMENT_REQUIREMENT: ["risk_assessment_attached"],
    FLAMMABLE_GAS_REQUIREMENT: ["flammable_gas_lel_percent"],
    OXYGEN_REQUIREMENT: [OXYGEN_COLUMN],
    COMBUSTIBLES_CLEARED_REQUIREMENT: ["combustibles_cleared_radius_m"],
    FIRE_WATCH_REQUIREMENT: ["fire_watch_minutes"],
    HYDROGEN_SULPHIDE_REQUIREMENT: ["hydrogen_sulphide_ppm"],
    CARBON_MONOXIDE_REQUIREMENT: ["carbon_monoxide_ppm"],
    RESCUE_PLAN_REQUIREMENT: ["rescue_plan_in_place"],
    PROVED_DEAD_REQUIREMENT: ["isolation_proved_dead"],
    PERSONAL_LOCKS_REQUIREMENT: ["personal_locks_applied"],
    CIRCUIT_EARTHS_REQUIREMENT: [SYSTEM_VOLTAGE_COLUMN, CIRCUIT_EARTHS_COLUMN],
    FALL_PROTECTION_REQUIREMENT: ["fall_protection_fitted"],
    SCAFFOLD_INSPECTION_REQUIREMENT: [ACCESS_METHOD_COLUMN, SCAFFOLD_INSPECTED_COLUMN],
    BARRIERS_REQUIREMENT: ["work_area_barriered"],
    WIND_SPEED_REQUIREMENT: [WORK_LOCATION_COLUMN, WIND_SPEED_COLUMN],
    SERVICES_LOCATED_REQUIREMENT: ["services_located"],
    TRENCH_SUPPORT_REQUIREMENT: [EXCAVATION_DEPTH_COLUMN, TRENCH_SUPPORT_COLUMN],
}

REQUIREMENTS_MET_COUNT_COLUMN = "requirements_met_count"
REQUIREMENTS_NOT_MET_COUNT_COLUMN = "requirements_not_met_count"
REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN = "requirements_not_documented_count"
REQUIREMENT_COUNT_RESULTS = {
    REQUIREMENTS_MET_COUNT_COLUMN: REQUIREMENT_MET,
    REQUIREMENTS_NOT_MET_COUNT_COLUMN: REQUIREMENT_NOT_MET,
    REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN: REQUIREMENT_NOT_DOCUMENTED,
}
CONFLICTING_PERMIT_TRIGGER_COLUMN = "conflicting_permit_trigger"
LIVE_PROCESS_TRIGGER_COLUMN = "live_process_trigger"
HIGH_VOLTAGE_TRIGGER_COLUMN = "high_voltage_trigger"
SIMOPS_TRIGGER_FLAG_COLUMNS = {
    CONFLICTING_PERMIT_TRIGGER: CONFLICTING_PERMIT_TRIGGER_COLUMN,
    LIVE_PROCESS_TRIGGER: LIVE_PROCESS_TRIGGER_COLUMN,
    HIGH_VOLTAGE_TRIGGER: HIGH_VOLTAGE_TRIGGER_COLUMN,
}


RANDOM_SEED = 42
SENTENCE_HELD_OUT_SHARE = 0.2
METRIC_DECIMALS = 4
MODEL_FILE_COMPRESSION = 3
PREPROCESSING_STEP_NAME = "preprocessing"
MODEL_STEP_NAME = "estimator"
NUMBER_FILL_STRATEGY = "median"
CATEGORY_FILL_STRATEGY = "most_frequent"

SUSPENSION_MODEL_FILE = "output/models/suspension_model.pkl"
WORK_HOURS_MODEL_FILE = "output/models/work_hours_model.pkl"
SENTENCE_MODEL_FILE = "output/models/sentence_model.pkl"
SUSPENSION_SCHEMA_FILE = "output/models/suspension_features.json"
WORK_HOURS_SCHEMA_FILE = "output/models/work_hours_features.json"
MODEL_METRICS_FILE = "output/model_metrics.json"

SUSPENSION_TARGET_ENCODING = {
    BOOLEAN_YES: 1,
    BOOLEAN_NO: 0,
}

SUSPENSION_NUMERIC_FEATURES = [
    "work_party_size", "planned_work_hours", "flammable_gas_lel_percent", OXYGEN_COLUMN,
    "hydrogen_sulphide_ppm", "carbon_monoxide_ppm", "combustibles_cleared_radius_m",
    "fire_watch_minutes", SYSTEM_VOLTAGE_COLUMN, "working_height_m", WIND_SPEED_COLUMN,
    EXCAVATION_DEPTH_COLUMN, CONCURRENT_LIVE_PERMITS_COLUMN, CONFLICTING_LIVE_PERMITS_COLUMN,
    REQUIREMENTS_MET_COUNT_COLUMN, REQUIREMENTS_NOT_MET_COUNT_COLUMN,
    REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN, CONFLICTING_PERMIT_TRIGGER_COLUMN,
    LIVE_PROCESS_TRIGGER_COLUMN, HIGH_VOLTAGE_TRIGGER_COLUMN,
]

SUSPENSION_CATEGORICAL_FEATURES = [
    PERMIT_TYPE_COLUMN, LINKED_PERMIT_TYPE_COLUMN, "work_activity", "area_classification",
    PROCESS_UNIT_STATUS_COLUMN, WORK_LOCATION_COLUMN, ACCESS_METHOD_COLUMN,
]

WORK_HOURS_NUMERIC_FEATURES = [
    "work_party_size", "fire_watch_minutes", SYSTEM_VOLTAGE_COLUMN, "working_height_m",
    EXCAVATION_DEPTH_COLUMN, WIND_SPEED_COLUMN, "planned_work_hours",
    CONCURRENT_LIVE_PERMITS_COLUMN, CONFLICTING_LIVE_PERMITS_COLUMN,
    CONFLICTING_PERMIT_TRIGGER_COLUMN, LIVE_PROCESS_TRIGGER_COLUMN, HIGH_VOLTAGE_TRIGGER_COLUMN,
]

WORK_HOURS_CATEGORICAL_FEATURES = [
    PERMIT_TYPE_COLUMN, LINKED_PERMIT_TYPE_COLUMN, "work_activity", "area_classification",
    PROCESS_UNIT_STATUS_COLUMN, WORK_LOCATION_COLUMN, ACCESS_METHOD_COLUMN, TRENCH_SUPPORT_COLUMN,
]

MINIMUM_SUSPENSION_ROC_AUC = 0.79
MINIMUM_WORK_HOURS_R2 = 0.84
MINIMUM_SENTENCE_MACRO_F1 = 0.95


OPEN_PERMIT_REQUESTS_FILE = "data/input/open_permit_requests.csv"
REQUEST_DOCUMENTS_FILE = "data/input/request_documents.csv"
LIVE_PERMIT_REGISTER_FILE = "data/input/live_permit_register.csv"
SUPPORTING_DOCUMENT_COUNT_COLUMN = "supporting_document_count"
LIVE_WHOLE_NUMBER_PATTERNS = {
    **FORM_WHOLE_NUMBER_PATTERNS,
    SUPPORTING_DOCUMENT_COUNT_COLUMN: WHOLE_NUMBER_PATTERN,
}
UNKNOWN_REQUEST_MESSAGE = "No open permit request has the reference {permit_reference}"

DOCUMENT_ID_COLUMN = "document_id"
DOCUMENT_TYPE_COLUMN = "document_type"
DOCUMENT_TEXT_COLUMN = "document_text"
DOCUMENT_TITLE_COLUMN = "document_title"
DOCUMENT_ID_PATTERN = r"^DOC-[0-9]{6}$"
DOCUMENT_VALUE_MAPPINGS = {
    "request_channel": REQUEST_CHANNEL_MAPPING,
    DOCUMENT_TYPE_COLUMN: DOCUMENT_TYPE_MAPPING,
}
DOCUMENT_IDENTIFIER_PATTERNS = {
    DOCUMENT_ID_COLUMN: DOCUMENT_ID_PATTERN,
}
DOCUMENT_WHOLE_NUMBER_PATTERNS = {}

PLANT_AREA_COLUMN = "plant_area"
EQUIPMENT_TAG_COLUMN = "equipment_tag"
PERMIT_STATUS_COLUMN = "permit_status"
PERMIT_STATUS_LIVE = "Live"
PERMIT_STATUS_SUSPENDED = "Suspended"
PERMIT_STATUS_MAPPING = {
    "live": PERMIT_STATUS_LIVE,
    "suspended": PERMIT_STATUS_SUSPENDED,
}
REGISTER_VALUE_MAPPINGS = {
    PLANT_AREA_COLUMN: PLANT_AREA_MAPPING,
    PERMIT_TYPE_COLUMN: PERMIT_TYPE_MAPPING,
    LINKED_PERMIT_TYPE_COLUMN: LINKED_PERMIT_TYPE_MAPPING,
    "work_activity": WORK_ACTIVITY_MAPPING,
    PERMIT_STATUS_COLUMN: PERMIT_STATUS_MAPPING,
}
REGISTER_REQUIRED_COLUMNS = [PERMIT_REFERENCE_COLUMN, PLANT_AREA_COLUMN, PERMIT_TYPE_COLUMN,
                             LINKED_PERMIT_TYPE_COLUMN, PERMIT_STATUS_COLUMN]
CONFLICTING_PERMIT_TYPES = [(HOT_WORK, CONFINED_SPACE_ENTRY), (HOT_WORK, EXCAVATION)]
BLANK_PLANT_AREA_MESSAGE = ("Request {permit_reference} gives no plant area, so the permits live "
                            "beside it cannot be counted")

STATEMENT_SENTENCE_PATTERN = r"(?<=[.!?])\s+(?=[A-Za-z0-9])"
NO_STATEMENT_TEXT_MESSAGE = "Document {document_id} has no text"
SENTENCE_TYPE_REQUIREMENT_KINDS = {
    ISOLATION: [PROVED_DEAD_REQUIREMENT, PERSONAL_LOCKS_REQUIREMENT, CIRCUIT_EARTHS_REQUIREMENT,
                SERVICES_LOCATED_REQUIREMENT],
    GAS_TESTING: [FLAMMABLE_GAS_REQUIREMENT, OXYGEN_REQUIREMENT, HYDROGEN_SULPHIDE_REQUIREMENT,
                  CARBON_MONOXIDE_REQUIREMENT],
    FIRE_PRECAUTIONS: [COMBUSTIBLES_CLEARED_REQUIREMENT, FIRE_WATCH_REQUIREMENT],
    RESCUE: [RESCUE_PLAN_REQUIREMENT],
    ACCESS: [FALL_PROTECTION_REQUIREMENT, SCAFFOLD_INSPECTION_REQUIREMENT, BARRIERS_REQUIREMENT,
             WIND_SPEED_REQUIREMENT, TRENCH_SUPPORT_REQUIREMENT],
    PERSONAL_PROTECTIVE_EQUIPMENT: [],
    SIMULTANEOUS_OPERATIONS: [],
    HOUSEKEEPING_AND_WASTE: [],
    BACKGROUND: [],
}
NO_REQUIREMENT_KINDS_MESSAGE = ("No requirements are configured for the sentence type "
                                "{sentence_type}")
STATEMENT_GAS_READING_PATTERNS = {
    "flammable_gas_lel_percent": r"(?i)([0-9]+(?:\.[0-9]+)?)% LEL",
    OXYGEN_COLUMN: r"(?i)([0-9]+(?:\.[0-9]+)?)% oxygen",
    "hydrogen_sulphide_ppm": r"(?i)([0-9]+(?:\.[0-9]+)?) ppm hydrogen sulphide",
    "carbon_monoxide_ppm": r"(?i)([0-9]+(?:\.[0-9]+)?) ppm carbon monoxide",
}
DISAGREES_WITH_FORM = "DISAGREES WITH FORM"
DESCRIBES_BLANK_ANSWER = "DESCRIBES BLANK ANSWER"

SCREENING_OF_ANOTHER_REQUEST_MESSAGE = ("The screening of {screening_reference} cannot be used for "
                                        "request {permit_reference}")
LIKELIHOOD_STEP = Decimal("0.01")
HIGH_LIKELIHOOD_MINIMUM = Decimal("0.40")
MEDIUM_LIKELIHOOD_MINIMUM = Decimal("0.20")
HIGH_LIKELIHOOD_BAND = "HIGH"
MEDIUM_LIKELIHOOD_BAND = "MEDIUM"
LOW_LIKELIHOOD_BAND = "LOW"
PLANNED_WORK_HOURS_COLUMN = "planned_work_hours"
EXTENSION_RATIO = Decimal("1.3")
LIKELIHOOD_DECIMALS = 2
WORK_HOURS_DECIMALS = 1
EXPECTED_HOURS_WITHOUT_PLAN_TEMPLATE = ("{expected_work_hours} work hours; the request gives no "
                                        "planned work hours, so no extension is worked out")
EXTENSION_TEMPLATE = ("{expected_work_hours} work hours against {planned_work_hours} planned: an "
                      "extension is expected")
NO_EXTENSION_TEMPLATE = ("{expected_work_hours} work hours against {planned_work_hours} planned: "
                         "no extension is expected")

SENTENCE_TYPE_GUIDE_DOCUMENTS = {
    ISOLATION: "Guide-Isolation.pdf",
    GAS_TESTING: "Guide-Gas-Testing.pdf",
    FIRE_PRECAUTIONS: "Guide-Fire-Precautions.pdf",
    RESCUE: "Guide-Rescue.pdf",
    ACCESS: "Guide-Access.pdf",
    PERSONAL_PROTECTIVE_EQUIPMENT: "Guide-Personal-Protective-Equipment.pdf",
    SIMULTANEOUS_OPERATIONS: "Guide-Simultaneous-Operations.pdf",
    HOUSEKEEPING_AND_WASTE: "Guide-Housekeeping-And-Waste.pdf",
}
NO_GUIDE_CONFIGURED_MESSAGE = "No statement guide is configured for {sentence_type}"
REVIEWER_CHECKS_SECTION_NUMBER = 3
REVIEWER_CHECKS_SECTION_TITLE = "What the reviewer checks"
DOCUMENTATION_SECTION_TITLE = "Documentation to provide"
PERMIT_DOCUMENT_SECTION_NUMBER = 7
PERMIT_DOCUMENT_SECTION_TITLE = "The permit document"
WORK_PARTY_BRIEFING_SECTION_NUMBER = 8
WORK_PARTY_BRIEFING_SECTION_TITLE = "The work party briefing"
CONTROL_OF_WORK_POLICY_DOCUMENT = "Control-Of-Work-Policy.pdf"
POLICY_SIMOPS_SECTION_NUMBER = 6
POLICY_SIMOPS_SECTION_TITLE = "The SIMOPS triggers"
POLICY_DECISIONS_SECTION_NUMBER = 7
POLICY_DECISIONS_SECTION_TITLE = "The three decisions"
POLICY_DOCUMENTS_SECTION_NUMBER = 8
POLICY_DOCUMENTS_SECTION_TITLE = "Documents and the SIMOPS brief"
SECTION_TITLE_MISMATCH_MESSAGE = ("Section {section_number} of {document_name} is not "
                                  "\"{section_title}\"")

PERMIT_DETAIL_COLUMNS = [PERMIT_TYPE_COLUMN, LINKED_PERMIT_TYPE_COLUMN, PLANT_AREA_COLUMN,
                         EQUIPMENT_TAG_COLUMN, "work_activity", "work_party_size",
                         PLANNED_WORK_HOURS_COLUMN]
PERMIT_CONTROL_COLUMNS = {
    HOT_WORK: ["flammable_gas_lel_percent", OXYGEN_COLUMN, "combustibles_cleared_radius_m",
               "fire_watch_minutes"],
    CONFINED_SPACE_ENTRY: ["flammable_gas_lel_percent", OXYGEN_COLUMN, "hydrogen_sulphide_ppm",
                           "carbon_monoxide_ppm", "rescue_plan_in_place"],
    ELECTRICAL_ISOLATION: [SYSTEM_VOLTAGE_COLUMN, "isolation_proved_dead", "personal_locks_applied",
                           CIRCUIT_EARTHS_COLUMN],
    WORK_AT_HEIGHT: ["working_height_m", ACCESS_METHOD_COLUMN, "fall_protection_fitted",
                     "work_area_barriered", WIND_SPEED_COLUMN],
    EXCAVATION: [EXCAVATION_DEPTH_COLUMN, "services_located", "work_area_barriered",
                 TRENCH_SUPPORT_COLUMN],
}
SIMOPS_BRIEF_DETAIL_COLUMNS = [PERMIT_TYPE_COLUMN, LINKED_PERMIT_TYPE_COLUMN, PLANT_AREA_COLUMN,
                               "area_classification", PROCESS_UNIT_STATUS_COLUMN,
                               WORK_LOCATION_COLUMN, "work_activity", EQUIPMENT_TAG_COLUMN,
                               "work_party_size"]

PROCESS_UNITS_GROUP = "Process units"
STORAGE_AREAS_GROUP = "Storage areas"
UTILITIES_GROUP = "Utilities"
NON_PROCESS_AREAS_GROUP = "Non-process areas"
AREA_GROUPS = {
    PROCESS_UNITS_GROUP: [REACTOR_HOUSE_A, REACTOR_HOUSE_B, DISTILLATION_UNIT, SOLVENT_RECOVERY,
                          CHLORINE_UNIT, CAUSTIC_EVAPORATION, DRUM_FILLING, FLARE_AREA,
                          EAST_PIPE_BRIDGE, WEST_PIPE_BRIDGE],
    STORAGE_AREAS_GROUP: [TANK_FARM_NORTH, TANK_FARM_SOUTH, ROAD_LOADING_BAY, RAIL_SIDINGS],
    UTILITIES_GROUP: [EFFLUENT_TREATMENT, BOILER_HOUSE, COOLING_TOWERS, COMPRESSOR_HOUSE,
                      MAIN_SUBSTATION, WATER_TREATMENT],
    NON_PROCESS_AREAS_GROUP: [SITE_ROADS_AND_DRAINS, ENGINEERING_WORKSHOPS, CENTRAL_WAREHOUSE,
                              LABORATORY, CONTROL_BUILDING, ADMINISTRATION_BLOCK, GATEHOUSE,
                              FIRE_STATION, WEIGHBRIDGE],
}
ROLE_AREA_AUTHORITY = "Area Authority"
DECISIONS = [DECISION_ISSUE, DECISION_RETURN, DECISION_REFER]
ISSUE_CLOSED_REASONS = {
    DECISION_RETURN: "a requirement is Not met or Not documented",
    DECISION_REFER: "a SIMOPS trigger applies",
}
DECISION_RECORD_FIELDS = ["decision", "decided_by", "decided_by_role", "decided_by_area_group",
                          "decision_reason"]
DECISION_FIELDS_MISSING_MESSAGE = ("The decision on request {permit_reference} gives no "
                                   "{fields}")
UNKNOWN_DECISION_MESSAGE = ("{decision} is not one of the decisions on request "
                            "{permit_reference}: {decisions}")
DECIDING_ROLE_MESSAGE = ("The role {role} may not decide request {permit_reference}: only an "
                         "{authority_role} decides a review")
NOT_APPOINTED_MESSAGE = ("{decided_by} is appointed for {area_group}, not for {plant_area}, "
                         "where request {permit_reference} is")
OWN_REVIEW_MESSAGE = ("{decided_by} opened the review of request {permit_reference} and may not "
                      "decide it")
DECISION_NOT_OPEN_MESSAGE = ("{decision} is not open for request {permit_reference}: "
                             "{reason}")


ORGANISATION_NAME = "Hexley Chemicals"
SITE_NAME = "Thornmere Works"
NOT_GIVEN_TEXT = "not given"
FORM_ANSWER_LABELS = {
    "request_channel": "the channel",
    REVISION_NUMBER_COLUMN: "the revision",
    PERMIT_TYPE_COLUMN: "the permit type",
    LINKED_PERMIT_TYPE_COLUMN: "the linked permit type",
    PLANT_AREA_COLUMN: "the plant area",
    "area_classification": "the hazardous area classification",
    PROCESS_UNIT_STATUS_COLUMN: "the process unit's status",
    WORK_LOCATION_COLUMN: "whether the work is indoors or outdoors",
    "work_activity": "the work activity",
    EQUIPMENT_TAG_COLUMN: "the equipment tag",
    "work_party_size": "the size of the work party",
    PLANNED_WORK_HOURS_COLUMN: "the planned work hours",
    "risk_assessment_attached": "whether a task risk assessment is attached",
    "flammable_gas_lel_percent": ("the flammable gas reading, in percent of the lower explosive "
                                  "limit"),
    OXYGEN_COLUMN: "the oxygen reading, in percent",
    "hydrogen_sulphide_ppm": "the hydrogen sulphide reading, in ppm",
    "carbon_monoxide_ppm": "the carbon monoxide reading, in ppm",
    "combustibles_cleared_radius_m": "the clearance of combustible materials, in metres",
    "fire_watch_minutes": "the fire watch after the work, in minutes",
    "rescue_plan_in_place": "whether a rescue plan is in place",
    SYSTEM_VOLTAGE_COLUMN: "the system voltage, in volts",
    "isolation_proved_dead": "whether the isolation is proved dead",
    "personal_locks_applied": "whether personal locks are applied",
    CIRCUIT_EARTHS_COLUMN: "whether circuit earths are applied",
    "working_height_m": "the working height, in metres",
    ACCESS_METHOD_COLUMN: "the access method",
    "fall_protection_fitted": "whether fall protection is fitted",
    SCAFFOLD_INSPECTED_COLUMN: "whether the scaffold is inspected and tagged",
    WIND_SPEED_COLUMN: "the wind at the work position, in metres per second",
    EXCAVATION_DEPTH_COLUMN: "the excavation depth, in metres",
    "services_located": "whether underground services are located",
    TRENCH_SUPPORT_COLUMN: "whether the sides of the trench are supported",
    "work_area_barriered": "whether the work area is barriered",
    CONCURRENT_LIVE_PERMITS_COLUMN: "the permits live in the plant area",
    CONFLICTING_LIVE_PERMITS_COLUMN: "the conflicting permits live in the plant area",
    REQUIREMENTS_MET_COUNT_COLUMN: "requirements Met",
    REQUIREMENTS_NOT_MET_COUNT_COLUMN: "requirements Not met",
    REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN: "requirements Not documented",
    CONFLICTING_PERMIT_TRIGGER_COLUMN: "the conflicting live permit trigger, 1 when it applies",
    LIVE_PROCESS_TRIGGER_COLUMN: "the live process unit trigger, 1 when it applies",
    HIGH_VOLTAGE_TRIGGER_COLUMN: "the high voltage trigger, 1 when it applies",
}
TRIGGER_ANSWER_COLUMNS = [PLANT_AREA_COLUMN, PROCESS_UNIT_STATUS_COLUMN, SYSTEM_VOLTAGE_COLUMN]
STATEMENT_GUIDE_ANSWER_COLUMNS = [PROCESS_UNIT_STATUS_COLUMN, WORK_LOCATION_COLUMN,
                                  "work_party_size", PLANNED_WORK_HOURS_COLUMN]
DECISION_DESCRIPTIONS = {
    DECISION_ISSUE: ("the work may go ahead as requested, with a permit document written to the "
                     "performing authority and then a work party briefing for the people doing "
                     "the work"),
    DECISION_RETURN: ("the request goes back to its performing authority with a rework request "
                      "naming everything to put right"),
    DECISION_REFER: ("the request goes to the SIMOPS panel with its review and a SIMOPS brief, "
                     "where the jobs in the area are kept apart and an exception can be argued"),
}
DOCUMENT_KINDS = {
    DECISION_ISSUE: "permit document",
    DECISION_RETURN: "rework request",
}
DOCUMENT_DETAIL_LABELS = {
    PERMIT_REFERENCE_COLUMN: "Permit reference",
    PERMIT_TYPE_COLUMN: "Permit type",
    LINKED_PERMIT_TYPE_COLUMN: "Linked permit type",
    PLANT_AREA_COLUMN: "Plant area",
    EQUIPMENT_TAG_COLUMN: "Equipment tag",
    "work_activity": "Work activity",
    "work_party_size": "Work party size",
    PLANNED_WORK_HOURS_COLUMN: "Planned work hours",
}
STANDARD_PASSAGE_TEMPLATE = "Under the {permit_type} standard: {passage}"
DOCUMENT_SIGN_OFF = f"Permit Office, {ORGANISATION_NAME}"
DOCUMENT_FORBIDDEN_TERMS = ["likelihood", "probability", "expected work hours"]
TEMPLATE_SLOT_CHARACTERS = ["[", "]"]
SIGN_OFF_SPACING_PATTERN = r"[\W_]+"
ROLE_PERMIT_COORDINATOR = "Permit Coordinator"
ROLE_CONTROL_ROOM_OPERATOR = "Control Room Operator"
SEED_USER_ACCOUNTS = [
    {"username": "dara", "display_name": "Dara Whitfield", "role": ROLE_AREA_AUTHORITY,
     "area_group": PROCESS_UNITS_GROUP, "password": "11111111"},
    {"username": "imogen", "display_name": "Imogen Hale", "role": ROLE_AREA_AUTHORITY,
     "area_group": PROCESS_UNITS_GROUP, "password": "11111111"},
    {"username": "kofi", "display_name": "Kofi Mensah", "role": ROLE_AREA_AUTHORITY,
     "area_group": STORAGE_AREAS_GROUP, "password": "11111111"},
    {"username": "ruth", "display_name": "Ruth Calder", "role": ROLE_AREA_AUTHORITY,
     "area_group": STORAGE_AREAS_GROUP, "password": "11111111"},
    {"username": "tomasz", "display_name": "Tomasz Nowak", "role": ROLE_AREA_AUTHORITY,
     "area_group": UTILITIES_GROUP, "password": "11111111"},
    {"username": "elena", "display_name": "Elena Varga", "role": ROLE_AREA_AUTHORITY,
     "area_group": UTILITIES_GROUP, "password": "11111111"},
    {"username": "gareth", "display_name": "Gareth Lloyd", "role": ROLE_AREA_AUTHORITY,
     "area_group": NON_PROCESS_AREAS_GROUP, "password": "11111111"},
    {"username": "sunita", "display_name": "Sunita Rao", "role": ROLE_AREA_AUTHORITY,
     "area_group": NON_PROCESS_AREAS_GROUP, "password": "11111111"},
    {"username": "marcus", "display_name": "Marcus Reed", "role": ROLE_PERMIT_COORDINATOR,
     "area_group": None, "password": "22222222"},
    {"username": "fiona", "display_name": "Fiona Blake", "role": ROLE_PERMIT_COORDINATOR,
     "area_group": None, "password": "22222222"},
    {"username": "hassan", "display_name": "Hassan Idris", "role": ROLE_PERMIT_COORDINATOR,
     "area_group": None, "password": "22222222"},
    {"username": "chloe", "display_name": "Chloe Mercer", "role": ROLE_PERMIT_COORDINATOR,
     "area_group": None, "password": "22222222"},
    {"username": "liam", "display_name": "Liam Doherty", "role": ROLE_CONTROL_ROOM_OPERATOR,
     "area_group": None, "password": "33333333"},
    {"username": "aisha", "display_name": "Aisha Khan", "role": ROLE_CONTROL_ROOM_OPERATOR,
     "area_group": None, "password": "33333333"},
    {"username": "nathan", "display_name": "Nathan Cole", "role": ROLE_CONTROL_ROOM_OPERATOR,
     "area_group": None, "password": "33333333"},
    {"username": "rosa", "display_name": "Rosa Delgado", "role": ROLE_CONTROL_ROOM_OPERATOR,
     "area_group": None, "password": "33333333"},
]

PERMIT_DATABASE_FILE = f"{DATABASE_DIRECTORY}/permitwise.db"
PERMIT_DATABASE_URL = f"sqlite:///./{PERMIT_DATABASE_FILE}"
CHECKPOINT_DATABASE_FILE = f"{DATABASE_DIRECTORY}/checkpoint.db"
USER_ACCOUNTS_TABLE = "user_accounts"
AGENT_RUNS_TABLE = "agent_runs"
PERMIT_REVIEWS_TABLE = "permit_reviews"
REVIEW_DECISIONS_TABLE = "review_decisions"
LLM_CALLS_TABLE = "llm_calls"
CHAT_TURNS_TABLE = "chat_turns"
RUN_ID_PREFIX = "RUN-"
REVIEW_ID_PREFIX = "REV-"
IDENTIFIER_HEX_LENGTH = 12
AGENT_RUN_STATUS_RUNNING = "RUNNING"
AGENT_RUN_STATUS_AWAITING_DECISION = "AWAITING_DECISION"
AGENT_RUN_STATUS_COMPLETE = "COMPLETE"
AGENT_RUN_STATUS_FAILED = "FAILED"
REVIEW_STATUS_AWAITING = "AWAITING_DECISION"
REVIEW_STATUS_ISSUED = "ISSUED"
REVIEW_STATUS_RETURNED = "RETURNED_TO_REQUESTER"
REVIEW_STATUS_REFERRED = "REFERRED_FOR_SIMOPS"
REVIEW_STATUS_BY_DECISION = {
    DECISION_ISSUE: REVIEW_STATUS_ISSUED,
    DECISION_RETURN: REVIEW_STATUS_RETURNED,
    DECISION_REFER: REVIEW_STATUS_REFERRED,
}

ENV_FILE = ".env"
GEMINI_API_KEY_VARIABLE = "GEMINI_API_KEY"
GEMINI_MODEL_VARIABLE = "GEMINI_MODEL"
GEMINI_MAX_OUTPUT_TOKENS = 8192
GEMINI_TEMPERATURE = 0.2
MILLISECONDS_PER_SECOND = 1000
AGENT_NAME_PLANNER = "review_planner_agent"
AGENT_NAME_REQUIREMENTS = "requirements_analyst_agent"
AGENT_NAME_EVIDENCE = "evidence_analyst_agent"
AGENT_NAME_RISK = "risk_analyst_agent"
AGENT_NAME_WRITER = "review_writer_agent"
AGENT_NAME_PERMIT_WRITER = "permit_writer_agent"
AGENT_NAME_BRIEFING_WRITER = "briefing_writer_agent"
AGENT_NAME_PERMIT_ASSISTANT = "permit_assistant_agent"
RESPONSE_DECORATION_CHARACTERS = "*#`_~ \t:"
RESPONSE_DECORATION_EDGE_PATTERN = r"^[\W_]+|[\W_]+$"


LLM_CALLS_BEFORE_DECISION = 5
LLM_CALLS_BY_DECISION = {
    DECISION_ISSUE: 7,
    DECISION_RETURN: 6,
    DECISION_REFER: 5,
}
REQUEST_STATUS_AWAITING_REVIEW = "AWAITING_REVIEW"
DECIDED_REVIEW_STATUSES = [REVIEW_STATUS_ISSUED, REVIEW_STATUS_RETURNED, REVIEW_STATUS_REFERRED]
REVIEW_STATUS_OPTIONS = [REVIEW_STATUS_AWAITING, REVIEW_STATUS_ISSUED, REVIEW_STATUS_RETURNED,
                         REVIEW_STATUS_REFERRED]
REQUEST_STATUS_LABELS = {
    REQUEST_STATUS_AWAITING_REVIEW: "Awaiting review",
    REVIEW_STATUS_AWAITING: "Awaiting a decision",
    REVIEW_STATUS_ISSUED: "Issued",
    REVIEW_STATUS_RETURNED: "Returned to requester",
    REVIEW_STATUS_REFERRED: "Referred for SIMOPS review",
}
REQUEST_STATUS_PHRASES = {
    REQUEST_STATUS_AWAITING_REVIEW: "awaiting review",
    REVIEW_STATUS_AWAITING: "awaiting a decision",
    REVIEW_STATUS_ISSUED: "issued",
    REVIEW_STATUS_RETURNED: "returned to requester",
    REVIEW_STATUS_REFERRED: "referred for SIMOPS review",
}
ALREADY_DECIDED_MESSAGE = "The review of request {permit_reference} has already been decided"
DECISION_IN_PROGRESS_MESSAGE = ("{pending_decided_by} has sent {pending_decision} for request "
                                "{permit_reference} and it is still being recorded; only "
                                "{pending_decided_by} can finish it, by sending the same decision "
                                "with the same reason again")
QUEUE_FIRST_PERMIT_TYPE = CONFINED_SPACE_ENTRY
QUEUE_PERMIT_TYPE_ALL = "All"
QUEUE_PERMIT_TYPE_OPTIONS = [QUEUE_PERMIT_TYPE_ALL, HOT_WORK, CONFINED_SPACE_ENTRY,
                             ELECTRICAL_ISOLATION, WORK_AT_HEIGHT, EXCAVATION]
COORDINATOR_REQUEST_MINIMUM_CHARACTERS = 5
COORDINATOR_REQUEST_MAXIMUM_CHARACTERS = 1000
DECISION_REASON_MINIMUM_CHARACTERS = 10
DECISION_REASON_MAXIMUM_CHARACTERS = 1000

ROLES_SEEING_EVERY_REVIEW = [ROLE_AREA_AUTHORITY, ROLE_PERMIT_COORDINATOR]
PAGE_QUEUE = "Request queue"
PAGE_DETAIL = "Request detail"
PAGE_DECISION = "Decision queue"
PAGE_AUDIT = "Audit trail"
PAGE_CHAT = "Permit chat"
ROLE_PAGE_ACCESS = {
    ROLE_AREA_AUTHORITY: [PAGE_QUEUE, PAGE_DETAIL, PAGE_DECISION, PAGE_AUDIT, PAGE_CHAT],
    ROLE_PERMIT_COORDINATOR: [PAGE_QUEUE, PAGE_DETAIL, PAGE_AUDIT, PAGE_CHAT],
    ROLE_CONTROL_ROOM_OPERATOR: [PAGE_CHAT],
}
PERMISSION_QUEUE = "queue"
PERMISSION_REVIEW = "review"
PERMISSION_DECISION = "decision"
PERMISSION_AUDIT = "audit"
PERMISSION_CHAT = "chat"
ROLE_PERMISSIONS = {
    ROLE_AREA_AUTHORITY: [PERMISSION_QUEUE, PERMISSION_REVIEW, PERMISSION_DECISION,
                          PERMISSION_AUDIT, PERMISSION_CHAT],
    ROLE_PERMIT_COORDINATOR: [PERMISSION_QUEUE, PERMISSION_REVIEW, PERMISSION_AUDIT,
                              PERMISSION_CHAT],
    ROLE_CONTROL_ROOM_OPERATOR: [PERMISSION_CHAT],
}
PERMISSION_BY_ROUTE = {
    ("GET", "/queue"): PERMISSION_QUEUE,
    ("GET", "/queue/{permit_reference}"): PERMISSION_QUEUE,
    ("POST", "/reviews"): PERMISSION_REVIEW,
    ("GET", "/reviews"): PERMISSION_AUDIT,
    ("POST", "/reviews/{review_id}/decision"): PERMISSION_DECISION,
    ("GET", "/reviews/{review_id}/calls"): PERMISSION_AUDIT,
    ("POST", "/chat/ask"): PERMISSION_CHAT,
    ("GET", "/chat/turns"): PERMISSION_CHAT,
    ("GET", "/audit/chat-turns"): PERMISSION_AUDIT,
}
USERNAME_WIDTH = 8
DISPLAY_NAME_WIDTH = 16

PASSWORD_HASH_ALGORITHM = "sha256"
PASSWORD_HASH_ROUNDS = 200000
PASSWORD_SALT_BYTES = 16
SESSION_TOKEN_BYTES = 24
SESSION_TOKEN_HEADER = "X-Session-Token"
SIGN_IN_FAILED_MESSAGE = "Username or password is not correct"
SIGN_IN_REQUIRED_MESSAGE = "Sign in first"
SESSION_ENDED_MESSAGE = ("Your session ended: this account has signed in somewhere else or "
                         "signed out. Sign in again to carry on.")
PERMISSION_REFUSED_MESSAGE = "The {role} role does not include the {permission} permission"
AREA_GROUP_REFUSED_MESSAGE = ("{username} is appointed for {area_group} and may not act on "
                              "request {permit_reference}, which is in {plant_area}")
REVIEW_NOT_FOUND_MESSAGE = "No permit review has that identifier"
REQUEST_ALREADY_REVIEWED_MESSAGE = ("Revision {revision_number} of request {permit_reference} has "
                                    "already been reviewed; each revision is reviewed once")
REVIEW_OUTSIDE_QUEUE_MESSAGE = ("Review {review_id} is of request {permit_reference}, which is not "
                                "in the request queue")
CHECKPOINT_MISSING_MESSAGE = "Review {review_id} has no workflow checkpoint in {checkpoint_file}"

CHAT_TYPE_PERMIT_STANDARD = "PERMIT STANDARD"
CHAT_TYPE_CONTROL_OF_WORK = "CONTROL OF WORK"
CHAT_TYPE_STATEMENT_GUIDANCE = "STATEMENT GUIDANCE"
CHAT_TYPE_REQUEST_STATUS = "REQUEST STATUS"
CHAT_TYPE_OUTCOME_PREDICTION = "OUTCOME PREDICTION"
CHAT_TYPE_PLANT_AREA_FIGURE = "PLANT AREA FIGURE"
CHAT_TYPE_OUT_OF_SCOPE = "OUT OF SCOPE"
CHAT_TYPE_OPTIONS = [CHAT_TYPE_PERMIT_STANDARD, CHAT_TYPE_CONTROL_OF_WORK,
                     CHAT_TYPE_STATEMENT_GUIDANCE, CHAT_TYPE_REQUEST_STATUS,
                     CHAT_TYPE_OUTCOME_PREDICTION, CHAT_TYPE_PLANT_AREA_FIGURE,
                     CHAT_TYPE_OUT_OF_SCOPE]
CHAT_ANSWERED_TYPES = [CHAT_TYPE_PERMIT_STANDARD, CHAT_TYPE_CONTROL_OF_WORK,
                       CHAT_TYPE_STATEMENT_GUIDANCE]
CHAT_REQUEST_TYPES = [CHAT_TYPE_REQUEST_STATUS, CHAT_TYPE_OUTCOME_PREDICTION]
CHAT_RECORD_TYPES = [CHAT_TYPE_REQUEST_STATUS, CHAT_TYPE_OUTCOME_PREDICTION,
                     CHAT_TYPE_PLANT_AREA_FIGURE]
CHAT_QUESTION_MINIMUM_CHARACTERS = 3
CHAT_QUESTION_MAXIMUM_CHARACTERS = 1000
CHAT_EXCLUDED_DOCUMENTS = ["Past-Permit-Decisions-Casebook.pdf"]
CHAT_RETRIEVAL_RESULT_COUNT = 4
CHAT_MAX_RELEVANCE_DISTANCE = 0.68
CHAT_HISTORY_TURN_LIMIT = 3
CHAT_PAGE_TURN_LIMIT = 20
CHAT_AUDIT_MAX_ROWS = 200
CHAT_NO_VALUE_REPLY = "none"
CHAT_FIRST_QUESTION_TEXT = "This is the first question in the conversation."
CHAT_RECORD_ANSWER_IN_HISTORY = "(answered from the permit records, which are not repeated here)"
PERMIT_REFERENCE_IN_TEXT_PATTERN = (r"(?i)(?<![0-9A-Za-z-])(?:PTW[0-9]{7}|PTW-[0-9]{1,7})"
                                    r"(?![0-9])")
NUMBER_IN_TEXT_PATTERN = r"(?<![0-9])0*{number}(?![0-9])"
NAME_IN_TEXT_PATTERN = r"\b{name}\b"
CHAT_PERMIT_TYPE_PATTERNS = {
    HOT_WORK: r"\bhot work\b",
    CONFINED_SPACE_ENTRY: r"\bconfined spaces?\b|\bentry\b|\bentries\b",
    ELECTRICAL_ISOLATION: r"\bisolations?\b",
    WORK_AT_HEIGHT: r"\bwork(?:ing)? at height\b",
    EXCAVATION: r"\bexcavations?\b|\bdig(?:s|ging)?\b",
}
CHAT_REFUSAL_MESSAGE = "The control of work documents do not answer that question."
OUTCOME_NOT_PREDICTED_LINE = ("A permit request is decided by an Area Authority, and the control "
                              "room is not told whether a permit is likely to be suspended or how "
                              "long its work will take.")
CHAT_NO_REQUEST_TEMPLATE = "No open permit request is recorded under reference {permit_reference}."
CHAT_REQUEST_NOT_NAMED_MESSAGE = ("Name the permit request by its reference, as PTW0041207, "
                                  "PTW-41207 or 41207.")
CHAT_LINKED_PERMIT_TEMPLATE = "{permit_type}, linked with {linked_permit_type}"
CHAT_OPERATOR_STATUS_TEMPLATE = "{permit_reference} ({permit_types}) in {plant_area} is {status}."
CHAT_ISSUED_WORK_TEMPLATE = "Equipment tag: {equipment_tag}. Work party size: {work_party_size}."
CHAT_NO_REVIEW_TEMPLATE = ("No review of request {permit_reference} has been recorded: it is "
                           "awaiting review.")
CHAT_REVIEW_TEMPLATE = ("{review_id}, request {permit_reference} revision {revision_number} "
                        "({permit_types}) in {plant_area}, opened by {coordinator_id}: {status}.")
CHAT_SCREENING_TEMPLATE = ("Assessment {assessment}. {requirements}. SIMOPS triggers: {triggers}. "
                           "The decisions open: {decisions_open}.")
CHAT_PERMITS_TEMPLATE = "Permits live in the plant area: {live_permits}."
CONFLICTING_PERMIT_TEMPLATE = "{permit_reference} (conflicting)"
CHAT_FIGURES_TEMPLATE = ("Suspension likelihood {suspension_likelihood} ({likelihood_band}). "
                         "Expected {work_hours_text}.")
CHAT_EVIDENCE_TEMPLATE = ("Statement evidence: {sentence_count} sentences read ({sentence_types}); "
                          "statement sentences flagged against the request: {flag_count}.")
CHAT_NO_DOCUMENTS_TEXT = "no document is attached"
CHAT_NONE_TEXT = "none"
CHAT_AWAITING_DECISION_TEMPLATE = ("It waits for an Area Authority's decision; the review "
                                   "recommends {recommended_decision}.")
CHAT_DECIDED_TEMPLATE = ("{decided_by}, {decided_by_role} for {decided_by_area_group}, decided "
                         "{decision}. The reason given: \"{decision_reason}\"")
GROUP_SIZE_MINIMUM = 5
GROUP_SIZE_LINE = ("Figures about permits in a plant area are shared only where at least five "
                   "permits make them up.")
CHAT_PLANT_AREA_FIGURE_TEMPLATE = ("{permit_count} {permit_type} permits have been worked in "
                                   "{plant_area}.")
CHAT_PLANT_AREA_NOT_NAMED_MESSAGE = ("Name one plant area, as Reactor House A, and one permit "
                                     "type: hot work, confined space entry, electrical isolation, "
                                     "work at height or excavation.")

import os

API_HOST = "0.0.0.0"
API_PORT = int(os.getenv("API_PORT", "8081"))

API_TITLE = "PermitWise AI API"

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    f"http://localhost:{API_PORT}"
)

API_REQUEST_TIMEOUT_SECONDS = 300

SERVICE_REQUIRED_FILES = [
    (CLEAN_PERMIT_HISTORY_TRAIN_FILE, "run_data_pipeline.py"),
    (CLEAN_PERMIT_HISTORY_EVAL_FILE, "run_data_pipeline.py"),
    (SUSPENSION_MODEL_FILE, "run_model_training.py"),
    (WORK_HOURS_MODEL_FILE, "run_model_training.py"),
    (SENTENCE_MODEL_FILE, "run_model_training.py"),
    (SUSPENSION_SCHEMA_FILE, "run_model_training.py"),
    (WORK_HOURS_SCHEMA_FILE, "run_model_training.py"),
    (OPEN_PERMIT_REQUESTS_FILE, "the supplied data folder"),
    (REQUEST_DOCUMENTS_FILE, "the supplied data folder"),
    (LIVE_PERMIT_REGISTER_FILE, "the supplied data folder"),
    (DOCS_DIRECTORY, "the supplied data folder"),
    (f"{VECTOR_STORE_DIRECTORY}/{VECTOR_STORE_FILE_NAME}", "run_knowledge_indexing.py"),
    (PERMIT_DATABASE_FILE, "run_database_setup.py"),
]

DASHBOARD_TITLE = "PermitWise AI"
DASHBOARD_SUBTITLE = f"{ORGANISATION_NAME} | Control of work at {SITE_NAME}"
DASHBOARD_ICON = ":material/construction:"
DASHBOARD_PAGE_STYLE = ('<style>input::-ms-reveal {display: none;} '
                        '[data-testid="stTextInput"] [data-testid="InputInstructions"] '
                        '{display: none;}</style>')
REVIEW_FAILED_MESSAGE = ("The review could not be completed: {reason}. No review was recorded, so "
                         "the request is still awaiting review and the review can be started "
                         "again.")
DECISION_FAILED_MESSAGE = ("The decision could not be finished: {reason}. Record the same decision "
                           "with the same reason again to finish it.")
CHAT_FAILED_MESSAGE = "The question could not be answered: {reason}. Ask it again."
PAGE_ICONS = {
    PAGE_QUEUE: ":material/format_list_numbered:",
    PAGE_DETAIL: ":material/assignment:",
    PAGE_DECISION: ":material/gavel:",
    PAGE_AUDIT: ":material/manage_search:",
    PAGE_CHAT: ":material/forum:",
}
PAGE_URL_PATHS = {
    PAGE_QUEUE: "request-queue",
    PAGE_DETAIL: "request-detail",
    PAGE_DECISION: "decision-queue",
    PAGE_AUDIT: "audit-trail",
    PAGE_CHAT: "permit-chat",
}
DECISION_BADGE_COLOURS = {DECISION_ISSUE: "green", DECISION_RETURN: "orange",
                          DECISION_REFER: "violet"}
STATUS_BADGE_COLOURS = {REQUEST_STATUS_AWAITING_REVIEW: "gray", REVIEW_STATUS_AWAITING: "blue",
                        REVIEW_STATUS_ISSUED: "green", REVIEW_STATUS_RETURNED: "orange",
                        REVIEW_STATUS_REFERRED: "violet"}
LIKELIHOOD_BAND_BADGE_COLOURS = {HIGH_LIKELIHOOD_BAND: "red", MEDIUM_LIKELIHOOD_BAND: "yellow",
                                 LOW_LIKELIHOOD_BAND: "green"}
CHAT_TYPE_BADGE_COLOURS = {CHAT_TYPE_PERMIT_STANDARD: "blue",
                           CHAT_TYPE_CONTROL_OF_WORK: "violet",
                           CHAT_TYPE_STATEMENT_GUIDANCE: "green",
                           CHAT_TYPE_REQUEST_STATUS: "orange",
                           CHAT_TYPE_OUTCOME_PREDICTION: "yellow",
                           CHAT_TYPE_PLANT_AREA_FIGURE: "primary",
                           CHAT_TYPE_OUT_OF_SCOPE: "gray"}
REQUIREMENT_RESULT_BADGE_COLOURS = {REQUIREMENT_MET: "green", REQUIREMENT_NOT_MET: "red",
                                    REQUIREMENT_NOT_DOCUMENTED: "orange"}
NEUTRAL_BADGE_COLOUR = "gray"
SENTENCE_TYPE_BADGE_COLOUR = "blue"
EXTENSION_COLOUR = "red"
LATENCY_SECONDS_DECIMALS = 1
