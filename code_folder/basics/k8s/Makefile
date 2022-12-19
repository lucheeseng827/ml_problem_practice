# This will allow the Pod associated with the delete-pods ServiceAccount to delete Pods in the other-namespace namespace.

poddeployment=nginx-pod
export poddeployment
# Create the ServiceAccount
kubectl create serviceaccount delete-pods

# Create the Role with the necessary permissions
kubectl create role delete-pods --verb=delete --resource=pods --namespace=other-namespace

# Bind the Role to the ServiceAccount
kubectl create rolebinding delete-pods --role=delete-pods --serviceaccount=default:delete-pods --namespace=other-namespace
