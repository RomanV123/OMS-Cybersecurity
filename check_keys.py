with open("check_keys.txt", "w") as file:
    for length in range(1,9):
        for number in range(10 ** length):
            key = str(number).zfill(length)
            file.write(key + "\n")
            
    
            