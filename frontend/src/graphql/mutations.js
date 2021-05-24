import gql from 'graphql-tag';

export const CreateMessage = gql`
  mutation($url: String!, $username: String!, $round: Int!) {
    createMessage(input: {
      url: $url
      username: $username
      round: $round
    }) {
      id 
      url
      username
      round
    }
  }
`