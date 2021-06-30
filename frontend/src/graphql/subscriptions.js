import gql from 'graphql-tag';

export const OnCreateMessage = gql`
  subscription($roomName: String) {
    onCreateMessage(roomName: $roomName) {
      id 
      roomName
      round
      url
      username
      ttl
    }
  }
`

export const OnCreateRoom = gql`
subscription {
  onCreateRoom {
    roomName
    ttl
  }
}
`

export const OnDeleteRoom = gql`
subscription($roomName: String) {
  onDeleteRoom(roomName: $roomName) {
    roomName
    ttl
  }
}
`