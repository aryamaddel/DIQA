import pyiqa

metric = pyiqa.create_metric("brisque")
print(metric("I02_01_03.png").item())
