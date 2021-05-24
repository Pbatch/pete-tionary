import gql from 'graphql-tag';

export const OnCreateMessage = gql`
  subscription {
    onCreateMessage {
      id 
      url
      username
      round
    }
  }
`