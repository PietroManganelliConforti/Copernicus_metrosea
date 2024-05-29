from torchvision import transforms

def get_augmentation(augmentations_test_number):

    augmentations = False #to check the input param
    
    if augmentations_test_number == "0": #default, augmentations_test_number = 0
        augmentations = None
    
    elif augmentations_test_number == "1": #0.4082017946,
        augmentations = transforms.Compose([
            transforms.ElasticTransform(p=0.5, alpha=1, sigma=0.07)])
    
    elif augmentations_test_number == "2": #0.3980821988
        augmentations = transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=None, shear=None, resample=False, fillcolor=0)])
    
    elif augmentations_test_number == "3": #0.4008300987
        augmentations = transforms.Compose([
            transforms.RandomResizedCrop(size=(224,224))])
    
    elif augmentations_test_number == "4": #0.402396974
        augmentations = transforms.Compose([
            transforms.GaussianNoise(p=0.5, var_limit=(10.0, 50.0))])
    
    elif augmentations_test_number == "5":
        augmentations = transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0),
                                     scale=None, shear=None, resample=False, fillcolor=0)])

    elif augmentations_test_number == "6":
        augmentations = transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.2, 0),
                                     scale=None, shear=None, resample=False, fillcolor=0)])
        
    elif augmentations_test_number == "7":
        augmentations = transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.05, 0),
                                     scale=None, shear=None, resample=False, fillcolor=0)])
    elif augmentations_test_number == "8":
        pass

    return augmentations