import pandas as pd
import json
import torch

# df1 = pd.read_parquet('/net/tscratch/people/plgjaneliasz/Lookback-Lens/data/new_version_merged_df.parquet')
# df2 = pd.read_parquet('/net/tscratch/people/plgjaneliasz/Lookback-Lens/data/cnndm.parquet')
# df3 = pd.read_parquet('/net/tscratch/people/plgjaneliasz/Lookback-Lens/data/xsum_df.parquet')


# df_combined = pd.concat([df1, df2, df3])

# print("df1.shape: ", df1.shape)
# print("df2.shape: ", df2.shape)
# print("df3.shape: ", df3.shape)
# print("df_combined.shape: ", df_combined.shape)

# df_combined = df_combined.drop_duplicates(subset=['id'], keep=False)

# print("after drop duplicated: ", df_combined.shape)

# df_combined.to_parquet('/net/tscratch/people/plgjaneliasz/Lookback-Lens/data/full_hallu_ds.parquet')

# df = pd.read_parquet('/net/tscratch/people/plgjaneliasz/Research/data/new_version_merged_df.parquet')

# df = df.drop_duplicates(subset=['id'], keep=False)

# df.to_parquet('/net/tscratch/people/plgjaneliasz/Research/data/new_version_merged_df.parquet')

# jsonfile1 = '/net/pr2/projects/plgrid/plggllmhallu/hallu/llama2_resps/cnndm/llama2_ll_orig_nq_clf/cnndm-lookback-decoding_combined.json'
# jsonfile2 = '/net/pr2/projects/plgrid/plggllmhallu/hallu/llama2_resps/our_ds/llama2_ll_orig_nq_clf/our_ds-lookback-decoding_combined.json'
# jsonfile3 = '/net/pr2/projects/plgrid/plggllmhallu/hallu/llama2_resps/xsum/llama2_ll_orig_nq_clf/xsum-lookback-decoding_combined.json'

# jsonfile_combined = '/net/pr2/projects/plgrid/plggllmhallu/hallu/llama2_resps/llama2_ll_orig_nq_clf/lookback-decoding_responses.json'

# with open(jsonfile1, 'r') as f:
#     data1 = json.load(f)

# with open(jsonfile2, 'r') as f:
#     data2 = json.load(f)

# with open(jsonfile3, 'r') as f:
#     data3 = json.load(f)

# data1.update(data2)
# data1.update(data3)

# with open(jsonfile_combined, 'w') as f:
#     json.dump(data1, f)

# print('Done')

# df1 = pd.read_json("/net/tscratch/people/plgjaneliasz/Lookback-Lens/outputs/anno-cnndm.jsonl", lines=True)
# df2 = pd.read_json("/net/tscratch/people/plgjaneliasz/Lookback-Lens/outputs/anno-our_ds.jsonl", lines=True)
# df3 = pd.read_json("/net/tscratch/people/plgjaneliasz/Lookback-Lens/outputs/anno-xsum.jsonl", lines=True)

# print("df1.shape: ", df1.shape)
# print("df2.shape: ", df2.shape)
# print("df3.shape: ", df3.shape)

# df_combined = pd.concat([df1, df2, df3])
# print("df_combined.shape: ", df_combined.shape)

# # save as jsonl
# df_combined.to_json('/net/tscratch/people/plgjaneliasz/Lookback-Lens/outputs/anno-full.jsonl', orient='records', lines=True)

list1 = torch.load("/net/tscratch/people/plgjaneliasz/Lookback-Lens/outputs/step01_cnndm.pt")
list2 = torch.load("/net/tscratch/people/plgjaneliasz/Lookback-Lens/outputs/step01_our_ds.pt")
list3 = torch.load("/net/tscratch/people/plgjaneliasz/Lookback-Lens/outputs/step01_xsum.pt")

print("list1: ", len(list1))
print("list2: ", len(list2))
print("list3: ", len(list3))

list_combined = list1 + list2 + list3

print("list_combined: ", len(list_combined))

torch.save(list_combined, "/net/tscratch/people/plgjaneliasz/Lookback-Lens/outputs/step01_full.pt")

