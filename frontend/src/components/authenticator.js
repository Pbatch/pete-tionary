import { AmplifyAuthenticator, AmplifySignUp } from '@aws-amplify/ui-react'

const Authenticator = () => (
  <AmplifyAuthenticator>
    <AmplifySignUp
      slot="sign-up"
      formFields={[
        { type: "username" },
        { type: "password" }
      ]}
    />
  </AmplifyAuthenticator>
)

export default Authenticator