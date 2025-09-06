import pyiqa

metric = pyiqa.create_metric("niqe")
print(metric("I02_01_03.png").item())
