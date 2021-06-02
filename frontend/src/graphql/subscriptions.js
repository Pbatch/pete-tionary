import gql from 'graphql-tag';

export const OnCreateMessage = gql`
  subscription {
    onCreateMessage {
      id 
      room
      round
      url
      username
    }
  }
`

export const OnCreateRoom = gql`
subscription {
  onCreateRoom {
    id 
    name
  }
}
`

export const OnDeleteRoom = gql`
subscription {
  onDeleteRoom {
    id 
    name
  }
}
`