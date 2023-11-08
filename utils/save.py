import os
import torch
import pickle
import json

def save(env_name, model_id, data, agent):
    # Create directory 'history' if it doesn't exist
    history_path = 'history'
    if not os.path.exists(history_path):
        os.makedirs(history_path)

    # Create subdirectory '<env_name>' inside of 'history' if it doesn't exist
    env_path = os.path.join(history_path, env_name)
    if not os.path.exists(env_path):
        os.makedirs(env_path)

    # Create subdirectory '<model_id>' inside of 'history/<env_name>' if it doesn't exist
    model_path = os.path.join(env_path, model_id)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Convert hyperparameters into json and save all files in the specified subdirectory
    data_json = json.dumps(data)
    weights_path = os.path.join(model_path, 'model_weights.pth')
    value_loss_path = os.path.join(model_path, 'value_loss_log.pkl')
    finish_step_path = os.path.join(model_path, 'finish_step_log.pkl')
    collected_reward_path = os.path.join(model_path, 'collected_reward_log.pkl')
    data_path = os.path.join(model_path, 'data.json')

    # Save the model weights
    torch.save(agent.act_net.state_dict(), weights_path)

    # Save the value loss log
    with open(value_loss_path, 'wb') as f:
        pickle.dump(agent.value_loss_log, f)

    # Save the finish step log
    with open(finish_step_path, 'wb') as f:
        pickle.dump(agent.finish_step_log, f)

    # Save the colected reward log
    with open(collected_reward_path, 'wb') as f:
        pickle.dump(agent.collected_reward_log, f)

    # Save the hyperparameters
    with open(data_path, 'w') as f:
        f.write(data_json)
