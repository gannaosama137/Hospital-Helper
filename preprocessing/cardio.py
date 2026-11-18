def clustering_preprocessing(df):

    # Select features for clustering
    cluster_features = ['gender', 'age', 'bmi', 'cholesterol', 'gluc',
                         'ap_hi', 'ap_lo' , 'smoke' , 'alco', 'active']


    # Remove ID and convert age from days to years
    df = df.drop(columns=['id'])
    df['age'] = (df['age'] / 365.25).round(1) 


    # Filter invalid medical measurements
    valid = (
        df['ap_hi'].between(70 , 250) & 
        df['ap_lo'].between(40 , 200) &
        (df['ap_hi'] > df['ap_lo'])   & 
        df['height'].between(120 , 220) &
        df['weight'].between(40 , 200)
    )
    print ("Percentage of rows removed: " , (df.shape[0] - valid.sum()) / df.shape[0] * 100 )
    df = df[valid].reset_index(drop=True)


    # Calculate BMI 
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df = df[df['bmi'].between(10, 50)]

    return df , cluster_features 