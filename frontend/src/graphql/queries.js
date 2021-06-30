import gql from 'graphql-tag';

export const ListMessages = gql`
  query($roomName: String!, $round: Int) {
    listMessages(roomName: $roomName, round: $round) {
      items {
        id 
        roomName
        round
        url
        username
        ttl
      }
    }
  }
`

export const ListRooms = gql`
  query {
    listRooms {
      items {
        roomName
        ttl
      }
    }
  }
`