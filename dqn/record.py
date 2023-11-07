from .helper import get_eval_agent
from ..callbacks.record import record

def dqn_record(env_name, saved_model_id):
    eval_agent = get_eval_agent(
        env_name=env_name,
        saved_model_id=saved_model_id
    )

    record(
        env_name=env_name,
        agent=eval_agent,
        saved_model_id=saved_model_id
    )