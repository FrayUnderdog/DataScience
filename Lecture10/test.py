with open("Airline_Satisfaction.csv", "rb") as f:
    for i in range(40):
        line = f.readline()
        print(i+1, line)
