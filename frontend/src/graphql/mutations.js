import gql from 'graphql-tag'

export const CreateMessage = gql`
  mutation($url: String!, $username: String!, $round: Int!, $roomName: String!) {
    createMessage(input: {
      roomName: $roomName
      round: $round
      url: $url
      username: $username
    }) {
      id 
      roomName
      round
      url
      username
    }
  }
`

export const CreateRoom = gql`
  mutation($roomName: String!) {
    createRoom(input: {
      roomName: $roomName
    }) {
      id 
      roomName
    }
  }
`

export const DeleteRoom = gql`
  mutation($id: ID!) {
    deleteRoom(id: $id) 
    {
      id 
    }
  }
`

export const DeleteMessage = gql`
  mutation($id: ID!) {
    deleteMessage(id: $id) 
    {
      id 
    }
  }
`