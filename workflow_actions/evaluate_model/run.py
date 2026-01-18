from workflow_actions.evaluate_model.evaluate_model import EvaluateModel
from workflow_actions.json_handlers import read_json_to_dict
from workflow_actions.paths import EVALUATE_MODEL_CONFIG_PATH

em_config = read_json_to_dict(EVALUATE_MODEL_CONFIG_PATH)
em = EvaluateModel(**em_config)
em.evaluate(k_triplets=8000)
