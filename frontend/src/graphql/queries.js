import gql from 'graphql-tag';

export const ListMessages = gql`
  query {
    listMessages {
      items {
        id 
        url
        username
        round
      }
    }
  }
`