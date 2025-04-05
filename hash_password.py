import streamlit_authenticator as stauth
passwords = ['password123', 'user2pass']  # Replace with your desired passwords
hashed_passwords = stauth.Hasher(passwords).generate()
print(hashed_passwords)