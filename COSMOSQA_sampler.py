import pandas as pd

path = '/home/felix/projects/research/datasets/COSMOSQA/{}.csv'

splits = ['train', 'valid', 'test']

# keywords = ['what may happen', 'what will happen']
before_keywords = ['what', 'before']
after_keywords = ['what', 'after']
if_keywords = ['what', 'if']

for split in splits:
    new_df = pd.DataFrame(columns=['id','context','question','answer0','answer1','answer2','answer3','label'])
    df = pd.read_csv(path.format(split))
    for row in df.iterrows():
        if all(keyword in row[1][2].lower() for keyword in if_keywords) or all(keyword in row[1][2].lower() for keyword in before_keywords) or all(keyword in row[1][2].lower() for keyword in after_keywords):
            new_row = pd.DataFrame({'id': row[1][0],
                                   'context': row[1][1],
                                    'question': row[1][2],
                                    'answer0': row[1][3],
                                    'answer1': row[1][4],
                                    'answer2': row[1][5],
                                    'answer3': row[1][6],
                                    'label': row[1][7]}, index=[0])
            frames = [new_df, new_row]
            new_df = pd.concat(frames)
    new_df.to_csv('/home/felix/projects/research/datasets/COSMOSQA/{}_temporal.csv'.format(split), index=False)
