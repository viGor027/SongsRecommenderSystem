import torch
from workflow_actions.prepare_dataset.source.spectogram_pipeline import SpectogramPipeline
from workflow_actions.prepare_dataset.source.constants import SPACE_DIR, LABELS_DIR, SONGS_DIR, SPEC_TYPE, N_MELS, N_SECONDS, STEP
from workflow_actions.prepare_dataset.source.utils import prepare_for_dataset
from workflow_actions.json_handlers import read_json_to_dict, write_dict_to_json
from cloud.cloud_utils import get_ready_model_from_gcs_checkpoint
import os

os.makedirs(SPACE_DIR, exist_ok=True)
model = get_ready_model_from_gcs_checkpoint(bucket_name='fine_tuned', folder_name='model_1',
                                            checkpoint_name='model', cfg_file_name='cfg')
model.eval()

title_to_multi_hot_mapping = read_json_to_dict(os.path.join(LABELS_DIR, 'labels.json'))
number_to_str_tag_mapping = read_json_to_dict(os.path.join(LABELS_DIR, 'mapping.json'))


def forward_conv_lstm(x):
    x = model.conv(x)
    x = x.permute(0, 2, 1)
    _, (h_n, _) = model.seq_encoder(x)
    x = h_n[-1]
    return x


def multi_hot_tags_to_str_tags(y):
    res = []
    for row in y:
        tmp = []
        for i, val in enumerate(row):
            if val == 1:
                tmp.append(number_to_str_tag_mapping[str(i)])
        res.append(tmp)
    return res


titles = []
specs = []
Y = []

ppl = SpectogramPipeline("")
ppl.set_config(
    n_mels=N_MELS,
    n_seconds=N_SECONDS,
    spec_type=SPEC_TYPE,
    step=STEP,
    validation_probability=0,
    labels_path=os.path.join(LABELS_DIR, 'labels.json')
)

batch_size = 64

for song in os.listdir(SONGS_DIR):
    data, _ = ppl.get_song_specs(
        song_path=os.path.join(SONGS_DIR, song),
        song_title=song[:-4],
        song_tags=title_to_multi_hot_mapping[song[:-4]]
    )

    if data is None:
        continue

    titles_temp, X_temp, Y_temp = prepare_for_dataset(data, shuffle=False, return_titles=True)
    X_temp, Y_temp = X_temp.float(), Y_temp.float()
    titles.extend(titles_temp)

    with torch.no_grad():
        for i in range(batch_size, len(X_temp), batch_size):
            specs.append(forward_conv_lstm(X_temp[i-batch_size:i]))
        specs.append(forward_conv_lstm(X_temp[(len(X_temp)//batch_size)*batch_size:]))

    Y.append(Y_temp)
    print("processed: ", song)

specs = torch.concat(specs, dim=0)
Y = torch.concat(Y, dim=0)


Y = Y.tolist()
Y = multi_hot_tags_to_str_tags(Y)
description = {'titles': titles, 'tags': Y}

write_dict_to_json(description, os.path.join(SPACE_DIR, 'space_index.json'))
torch.save(specs, os.path.join(SPACE_DIR, 'space.pt'))
