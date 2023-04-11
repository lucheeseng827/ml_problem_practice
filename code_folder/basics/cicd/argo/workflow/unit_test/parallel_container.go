package (
    "context"
    "fmt"
    "testing"
    "time"

    "github.com/argoproj/argo-workflows/v3/pkg/apiclient/workflow"
    "github.com/argoproj/argo-workflows/v3/pkg/apis/workflow/v1alpha1"
    "github.com/argoproj/argo-workflows/v3/pkg/client/clientset/versioned"
    "github.com/stretchr/testify/assert"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/client-go/rest"
    "k8s.io/client-go/tools/clientcmd"
)

func TestWorkflow(t *testing.T) {
    // Load the Kubernetes config
    kubeconfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
        clientcmd.NewDefaultClientConfigLoadingRules(),
        &clientcmd.ConfigOverrides{},
    )
    config, err := kubeconfig.ClientConfig()
    assert.NoError(t, err)

    // Create the Argo Workflow clientset
    clientset, err := versioned.NewForConfig(config)
    assert.NoError(t, err)

    // Create the Argo Workflow service client
    serviceClient := workflow.NewServiceClient(config)

    // Define the workflow manifest
    wf := v1alpha1.Workflow{
        ObjectMeta: metav1.ObjectMeta{
            GenerateName: "test-workflow-",
        },
        Spec: v1alpha1.WorkflowSpec{
            Entry: "main",
            Templates: []v1alpha1.Template{
                {
                    Name: "main",
                    Container: &v1alpha1.Container{
                        Image:   "my-image",
                        Command: []string{"echo", "Hello, world!"},
                    },
                },
            },
        },
    }

    // Submit the workflow to the Kubernetes cluster
    _, err = clientset.ArgoprojV1alpha1().Workflows("default").Create(context.Background(), &wf, metav1.CreateOptions{})
    assert.NoError(t, err)

    // Wait for the workflow to complete
    for {
        w, err := serviceClient.GetWorkflow(context.Background(), &workflow.GetWorkflowRequest{
            Name:      wf.ObjectMeta.Name,
            Namespace: wf.ObjectMeta.Namespace,
        })
        assert.NoError(t, err)

        if w.Status.Phase.Completed() {
            // Workflow completed successfully
            assert.Equal(t, v1alpha1.WorkflowSucceeded, w.Status.Phase)
            break
        }

        if w.Status.Phase.Failed() {
            // Workflow failed
            t.Errorf("Workflow failed: %s", w.Status.Phase)
            break
        }

        time.Sleep(1 * time.Second)
    }
}
