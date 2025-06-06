import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil
from PIL import Image

def remove_duplicates(df):
    return df.drop_duplicates(inplace=False)

def order_coordinates(df):
    df = df.copy()
    df['x1'], df['x2'] = df[['x1', 'x2']].min(axis=1), df[['x1', 'x2']].max(axis=1)
    df['y1'], df['y2'] = df[['y1', 'y2']].min(axis=1), df[['y1', 'y2']].max(axis=1)
    return df

def adjust_no_area_boxes(df, width_height):
    df = df.copy()
    add_per_dim = width_height / 2
    
    same_y = df['y1'] == df['y2']
    df.loc[same_y, 'y1'] = df.loc[same_y, 'y1'] - add_per_dim
    df.loc[same_y, 'y2'] = df.loc[same_y, 'y2'] + add_per_dim

    same_x = df['x1'] == df['x2']
    df.loc[same_x, 'x1'] = df.loc[same_x, 'x1'] - add_per_dim
    df.loc[same_x, 'x2'] = df.loc[same_x, 'x2'] + add_per_dim

    return df

def remove_dot_boxes(df):
    df = df.copy()
    return df[~((df['x1'] == df['x2']) & (df['y1'] == df['y2']))]


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

def resize_images(img_folder, new_size: tuple):

    for filename in os.listdir(img_folder):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(img_folder, filename)
            img = Image.open(img_path)
            img_resized = img.resize(new_size)
            img_resized.save(img_path)

def get_image_size_dict(folder_path):
    image_size_dict = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            img = Image.open(os.path.join(folder_path, filename))
            image_size_dict[filename] = img.size

    return image_size_dict

def add_image_size_to_df(df, image_size_dict):
    df = df.copy()
    df['original_img_width'] = df['image'].map(lambda img: image_size_dict[img][0])
    df['original_img_height'] = df['image'].map(lambda img: image_size_dict[img][1])
    return df

def copy_imgs_to_folder(df, dst_folder, org_img_folder_path):
    for _, row in df.iterrows():
        img_filename = row['image']
        img_type = row['type']
        src_path = org_img_folder_path + '/images_' + img_type + '/' + img_filename
        
        shutil.copy(src_path, dst_folder + '/' + img_filename)


def prepare_bboxes(df):
    df = df.copy()
    df["x1"] = df["x1"] / df["original_img_width"]
    df["x2"] = df["x2"] / df["original_img_width"]
    df["y1"] = df["y1"] / df["original_img_height"]
    df["y2"] = df["y2"] / df["original_img_height"]

    df["x_center"] = (df["x1"] + df["x2"]) / 2
    df["y_center"] = (df["y1"] + df["y2"]) / 2
    df["bb_width"] = df["x2"] - df["x1"]
    df["bb_height"] = df["y2"] - df["y1"]
    
    return df

def store_lables_as_txt(df, output_path):
    os.makedirs(output_path, exist_ok=True)

    for image_name, group in df.groupby("image"):
        lines = []

        for _, row in group.iterrows():
            line = f"0 {row['x_center']:.6f} {row['y_center']:.6f} {row['bb_width']:.6f} {row['bb_height']:.6f}"
            lines.append(line)

        filename = os.path.splitext(os.path.basename(image_name))[0] + ".txt"
        filepath = os.path.join(output_path, filename)

        with open(filepath, "w") as f:
            f.write("\n".join(lines))


def train_val_test_split(df, train_size, val_size=None, random_state=42, use_val=True):

    if use_val:
        if val_size is None:
            raise ValueError("val_size must be specified when use_val is True.")
        
        val_test_size = round(1 - train_size, 5)
        train_df, temp_df = train_test_split(
            df,
            test_size=val_test_size,
            stratify=df['type'],
            random_state=random_state
        )

        test_size_prop = round((1 / val_test_size) * (val_test_size - val_size), 5)

        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_size_prop,
            stratify=temp_df['type'],
            random_state=random_state
        )
        return train_df, val_df, test_df
    else:
        test_size = round(1 - train_size, 5)
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['type'],
            random_state=random_state
        )
        return train_df, test_df


def check_type_ratio(train_df, eval_df = None, test_df = None):
    train_rows = train_df.shape[0]
    train_boom = train_df[train_df['type'] == 'boom'].shape[0]
    train_drone = train_df[train_df['type'] == 'drone'].shape[0]
    train_handheld = train_df[train_df['type'] == 'handheld'].shape[0]

    print(f'------TRAIN DATA:------')
    print(f'Boom portion: {(100 / train_rows) * train_boom}%')
    print(f'Drone portion: {(100 / train_rows) * train_drone}%')
    print(f'Handheld portion: {(100 / train_rows) * train_handheld}%')

    if eval_df is not None:
        eval_rows = eval_df.shape[0]
        eval_boom = eval_df[eval_df['type'] == 'boom'].shape[0]
        eval_drone = eval_df[eval_df['type'] == 'drone'].shape[0]
        eval_handheld = eval_df[eval_df['type'] == 'handheld'].shape[0]
    
        print(f'------EVALUATION DATA:------')
        print(f'Boom portion: {(100 / eval_rows) * eval_boom}%')
        print(f'Drone portion: {(100 / eval_rows) * eval_drone}%')
        print(f'Handheld portion: {(100 / eval_rows) * eval_handheld}%')

    if test_df is not None:
        test_rows = test_df.shape[0]
        test_boom = test_df[test_df['type'] == 'boom'].shape[0]
        test_drone = test_df[test_df['type'] == 'drone'].shape[0]
        test_handheld = test_df[test_df['type'] == 'handheld'].shape[0]
    
        print(f'------TEST DATA:------')
        print(f'Boom portion: {(100 / test_rows) * test_boom}%')
        print(f'Drone portion: {(100 / test_rows) * test_drone}%')
        print(f'Handheld portion: {(100 / test_rows) * test_handheld}%')
        
    













    
