import os
import torch
import pickle
import json
import uuid

def save(env_name, model_name, hyperparameters, agent):
    model_id = str(uuid.uuid4())

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
    hyperparameters_json = json.dumps(hyperparameters)
    weights_path = os.path.join(model_path, f'{model_name}_model_weights_{model_id}.pth')
    value_loss_path = os.path.join(model_path, f'{model_name}_value_loss_log_{model_id}.pkl')
    finish_step_path = os.path.join(model_path, f'{model_name}_finish_step_log_{model_id}.pkl')
    hyperparameters_path = os.path.join(model_path, f'{model_name}_hyperparameters_{model_id}.json')

    # Save the model weights
    torch.save(agent.act_net.state_dict(), weights_path)

    # Save the value loss log
    with open(value_loss_path, 'wb') as f:
        pickle.dump(agent.value_loss_log, f)

    # Save the finish step log
    with open(finish_step_path, 'wb') as f:
        pickle.dump(agent.finish_step_log, f)

    # Save the hyperparameters
    with open(hyperparameters_path, 'w') as f:
        f.write(hyperparameters_json)
