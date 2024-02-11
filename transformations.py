import albumentations as A

#pipeline di trasformazioni con Albumentations - dà errori nell'applicarlo a tutta sequenza

TRANSFORMATIONS = {
    "train": A.Compose([
        A.VerticalFlip(), 
        A.RandomScale(scale_limit=0.3),
        A.RandomBrightness(limit = (0.5,3)),  #volendo c'è anche RandomBrightnessContrast
        A.Affine(shear=0.3)
    ])
}