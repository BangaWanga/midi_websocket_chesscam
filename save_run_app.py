import os

errorlist = []

#Choose program-type

while True:
    print("Do you want to create a server and send midi data or do you want to create a client and sync with a midi-device?")
    mode =input("Choose (1) or (2)\n")
    if mode=="1":
        script = ("py server.py")
        print("Creating server...")
        break
    elif mode=="2":
        print("Creating client...")
        script = ("py client.py")
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



