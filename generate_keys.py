with open("generate_keys.txt", "w") as file:
    for number in range(10 ** 8):
        key = str(number).zfill(8)
        file.write(key + "\n")
            