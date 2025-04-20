def remove_duplicates(df):
    return df.drop_duplicates(inplace=False)

def order_coordinates(df):
    df = df.copy()
    df['x1'], df['x2'] = df[['x1', 'x2']].min(axis=1), df[['x1', 'x2']].max(axis=1)
    df['y1'], df['y2'] = df[['y1', 'y2']].min(axis=1), df[['y1', 'y2']].max(axis=1)
    return df

def remove_no_area_boxes(df):
    df = df.copy()
    return df[(df['x1'] != df['x2']) & (df['y1'] != df['y2'])]


def unify_img_suffix(folder_path):
    rename_extensions = ['.JPG', '.Jpeg']

    for filename in os.listdir(folder_path):
        name, ext = os.path.splitext(filename)

        if ext in rename_extensions:
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, name + '.jpg')
            os.rename(old_path, new_path)
            print(f'Renamed: {filename} -> {name}.jpg')

def unify_img_suffix_df(df):
    df = df.copy()
    df['image'] = df['image'].str.replace(r'\.(jpe?g)$', '.jpg', case=False, regex=True)
    return df

def clip_negative_coord_values(df):
    df = df.copy()
    cols = ['x1', 'y1', 'x2', 'y2']
    df[cols] = df[cols].clip(lower=0)
    return df