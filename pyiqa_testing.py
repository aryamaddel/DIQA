import pyiqa

metric = pyiqa.create_metric("brisque")
print(metric("koniq10k_512x384/826373.jpg").item())


metric = pyiqa.create_metric("hyperiqa")
print(metric("koniq10k_512x384/826373.jpg").item())


metric = pyiqa.create_metric("maniqa")
print(metric("koniq10k_512x384/826373.jpg").item())


metric = pyiqa.create_metric("niqe")
print(metric("koniq10k_512x384/826373.jpg").item())


metric = pyiqa.create_metric("niqe")
print(metric("koniq10k_512x384/826373.jpg").item())
