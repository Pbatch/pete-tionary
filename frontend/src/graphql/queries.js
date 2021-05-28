import gql from 'graphql-tag';

export const ListMessages = gql`
  query {
    listMessages {
      items {
        id 
        room
        round
        url
        username
      }
    }
  }
`

export const ListRooms = gql`
  query {
    listRooms {
      items {
        id 
        name
      }
    }
  }
`