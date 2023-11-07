from .helper import get_eval_agent
from ..callbacks.eval import eval

def dqn_eval(env_name, saved_model_id):
    eval_agent = get_eval_agent(
        env_name=env_name,
        saved_model_id=saved_model_id
    )

    eval(
        env_name=env_name,
        agent=eval_agent
    )