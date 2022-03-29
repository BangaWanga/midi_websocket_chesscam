import os

errorlist = []

#Choose program-type

while True:

    print("Do you want to create a server and send midi data or do you want to create a client and sync with a midi-device?")
    mode = input("Choose (1) or (2)\n")
    if mode == "1":
        print("Creating server...")
        script = ("py Server\server.py")
        break
    elif mode == "2":
        print("Creating Client...")
        script = ("py Client\client_alt.py")
        break
    else:
        print("Incorrect input...")

while True:
    try:
        os.system(script)

    except Exception as exception:

        print("Client failed. Restarting")
        errorlist.append(type(exception).__name__)
        print(f"Errors: {errorlist}")



