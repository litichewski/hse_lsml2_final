import neptune
import torch
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel

APIKEY = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmOTA2YTU3MS03M2Q4LTQ2NWItOGRjMS1kY2YxMTU1MmMzMDkifQ=="

##
model = neptune.init_model(
     with_id="LSMLFIN-MOD",project='litichewski/lsml-final', api_token=APIKEY,
 )

model_versions_df = model.fetch_model_versions_table().to_pandas()
production_models = model_versions_df[
    model_versions_df["sys/stage"] == "production"
]
prod_id = production_models['sys/id'].iloc[0]
model = neptune.init_model_version(with_id = prod_id, project="litichewski/lsml-final",api_token=APIKEY)
model["model"].download("./model.pt")
model.stop()

model_loaded = GPT2LMHeadModel.from_pretrained('gpt2')

state_dict = torch.load('model.pt')
##

model_loaded.load_state_dict(state_dict)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")