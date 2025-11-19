import datasets

ds = datasets.load_from_disk("/root/autodl-fs/test/train")

print(ds[0])