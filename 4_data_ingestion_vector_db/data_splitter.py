import pandas as pd
import json

df = pd.read_csv("podcastdata_dataset.csv")
df.head()

for i in range(len(df)):
    row = df.iloc[i]
    guest = row["guest"]
    title = row["title"]
    guest = guest.replace("/", "_")
    title = title.replace("/", "_")
    file_name = f"{guest}_{title}"
    out_dict = {
        "title": title,
        "guest": guest,
        "text": row["text"],
    }

    # save as json

    with open(
        f"/data/{file_name}.json",
        "w",
    ) as f:
        json.dump(out_dict, f)
