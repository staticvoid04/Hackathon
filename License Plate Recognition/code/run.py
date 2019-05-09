import infer
import gen
import train
import sys
import glob

test = "../cars/test_080.jpg"
GWAGEN = "../cars/eu2.jpg"
HF99VY = "../cars/SA_1.jpg"
test = "../cars/SA_6.jpg"
TS260AK = "../cars/test_005.jpg"
RK828AG = "../cars/test_007.jpg"
RK346AL = "../cars/test_011.jpg"
RK297AT = "../car/test_012.jpg"
RK857AL = "../cars/test_013.jpg"
RKO26AJ = "../cars/test_024.jpg"
RK896AO = "../cars/test_025.jpg"
RK492A0 = "../cars/test_026.jpg"
RK819AN = "../cars/test_030.jpg"
RK767A0 = "../cars/test_047.jpg"
RKT67AG = "../cars/test_048.jpg"
RK30OAG = "../cars/test_056.jpg"
RK300AS = "../cars/test_057.jpg"
RK485AF = "../cars/test_058.jpg"
RKQ69AV = "../cars/test_094.jpg"

# Arguements that can be passed through when running run.py
def main():
    args = sys.argv
    if(len(args) > 1):
        if("gen" in args):
            gen.main()
        if("train" in args):
            train.main()
        if("infer" in args):
            plate = infer.KNN(RK828AG)
            print("Prediction")
            print(plate)
        if("all" in args):
            gen.main()
            train.main()
            plate = infer.KNN(RK828AG)
            print("Prediction")
            print(plate)
        if("test"in args):
            test_cars()
            return


def test_cars():
    prev = ''
    cars = glob.glob("../cars/*.jpg")
    for car in cars:
        plate = infer.KNN(car)
        if(prev != plate):
            print(plate, "=", car)
        prev = plate


if __name__ == "__main__":
    main()
