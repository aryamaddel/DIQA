import pyiqa

metric = pyiqa.create_metric("piqe")
print(metric("I02_01_03.png").item())
