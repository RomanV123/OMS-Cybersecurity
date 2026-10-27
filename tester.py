
import hashlib

def obscure_input(input_value):
    encoded_input = input_value.encode()
    return hashlib.sha256(encoded_input).hexdigest()

def obscure_constant():
    secret_key = b"_CS6035-Kust0-Qu3rYL33t"
    return hashlib.sha256(secret_key).hexdigest()

def main():
    print("==== You found Flag 7! ====")
    
    # Input prompt for GTID
    u_input = input("Enter your GTID : ").strip()

    if not u_input:
        print("Error: GTID cannot be empty.")
        return

    # Obfuscated hash generation
    first_hash = obscure_input(u_input)
    second_hash = obscure_constant()
    
    # Concatenating the hashes directly
    combined = first_hash + second_hash

    # Displaying the final combined hash
    print("Combined hash   :  ", combined)

if __name__ == "__main__":
    main()
