from dataset import dataset
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):
    train_set = dataset.LandslideDataSet(args.data_dir, args.train_list,
                                         transform=None,
                                         set_mask='masked')
    test_set = dataset.LandslideDataSet(args.data_dir, args.test_list, set_mask='masked')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader
