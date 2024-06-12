from torchvision import transforms
from torchvision.transforms import v2

def get_augmentation(augmentations_test_number):

    augmentations = False #to check the input param
    
    if augmentations_test_number == "0": #default, augmentations_test_number = 0
        augmentations = None

    elif augmentations_test_number == "1":
        
        augmentations = transforms.Compose([
            v2.GaussianBlur(kernel_size=(3,3))
        ])

    elif augmentations_test_number == "2": #0.3980821988
        augmentations = transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=None, shear=None)])
    
    elif augmentations_test_number == "3": #0.4008300987
        augmentations = transforms.Compose([
            transforms.RandomResizedCrop(size=(224,224))])
    
    elif augmentations_test_number ==  "4":
        augmentations = transforms.Compose([
            lambda x: x + torch.normal(0, 0.1, size=x.shape)
        ])
     
    elif augmentations_test_number == "5":
        augmentations = transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0),
                                     scale=None, shear=None)])

    elif augmentations_test_number == "6":
        augmentations = transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.2, 0),
                                     scale=None, shear=None)])
        
    elif augmentations_test_number == "7":
        augmentations = transforms.Compose([
            transforms.RandomAffine(degrees=0, translate=(0.05, 0),
                                     scale=None, shear=None)])
        
    elif augmentations_test_number ==  "8":
        
        augmentations = transforms.Compose([
            v2.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random', inplace=False)
        ])

    elif augmentations_test_number == "9":
        
        augmentations = transforms.Compose([
            v2.RandomCrop(size=(220,220)),
            v2.Resize(size=(224,224))
        ])

        

    return augmentations



if __name__ == "__main__":

    from dataset2D import Dataset_2D_copernicus, merge_2D_dataset
    import torch

    augs = [(i,get_augmentation(str(i))) for i in range(1,9)]   

    dataset_2D = merge_2D_dataset(folder_path = "2D_Dataset_copernicus_only_tensors/",
                                    pred_label_lat = "45.60", pred_label_lon = "13.54",
                                    transforms = 0)
    sample = dataset_2D[0][0]

    aug_samples= []


    for i, (n,aug)in enumerate(augs):
        augmented_sample = aug(sample)
        print(i,n, torch.equal(sample,augmented_sample) , augmented_sample.shape)
    