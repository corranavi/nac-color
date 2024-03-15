
datagen_norm = ImageDataGenerator(samplewise_center=True,
                                          samplewise_std_normalization=True)

X_val = normalize_test(datagen_norm, X_val)

"""
    normalize_test is used to apply z-normalization (0 mean and 1 std) to the validation and test set (this normalization step is done by the ImageDataGenerator for augmentation on the training set)
"""
#Questo lo posso evitare se normalizzo ciascuna slice quando vado a caricarla - istanziare il Dataset
def normalize_test(datagen, X_train):
    X_aug = []
    for gen_X in X_train:
        for i in range(len(gen_X)):
            gen_X[i] = datagen.standardize(gen_X[i])
        X_aug.append(gen_X)
    return X_aug