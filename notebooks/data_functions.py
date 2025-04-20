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

def resize_images(input_folder, output_folder, new_size: tuple):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            img_resized = img.resize(new_size)
            img_resized.save(os.path.join(output_folder, filename))

def get_image_size_dict(folder_path):
    image_size_dict = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            img = Image.open(os.path.join(folder_path, filename))
            image_size_dict[filename] = img.size

    return image_size_dict

def add_image_size_to_df(df, image_size_dict):
    df = df.copy()
    df['original_width'] = df['image'].map(lambda img: image_size_dict[img][0])
    df['original_height'] = df['image'].map(lambda img: image_size_dict[img][1])
    return df

def copy_imgs_to_folder(df, dst_folder, org_img_folder_path):
    for index, row in df.iterrows():
        img_filename = row['image']
        img_type = row['type']
        src_path = org_img_folder_path + '/images_' + img_type
        
        shutil.copy(src_path, dst_folder + '/' + image)
