import jinja2


def generate_crd(
    resource_name, group, version, scope, plural, singular, kind, shortname
):
    # Load the template
    with open("crd_template.yaml.j2", "r") as f:
        template_content = f.read()
        template = jinja2.Template(template_content)

    # Render the template with the provided variables
    rendered_content = template.render(
        resource_name=resource_name,
        group=group,
        version=version,
        scope=scope,
        plural=plural,
        singular=singular,
        kind=kind,
        shortname=shortname,
    )

    # Save or print the rendered content
    with open(f"{kind.lower()}_crd.yaml", "w") as f:
        f.write(rendered_content)


# Example usage
generate_crd(
    "exampleresources",
    "sample.k8s.io",
    "v1",
    "Namespaced",
    "exampleresources",
    "exampleresource",
    "ExampleResource",
    "er",
)


"""
crd_template.yaml.j2
---
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: {{ resource_name }}.{{ group }}
spec:
  group: {{ group }}
  versions:
    - name: {{ version }}
      served: true
      storage: true
  scope: {{ scope }}
  names:
    plural: {{ plural }}
    singular: {{ singular }}
    kind: {{ kind }}
    shortNames:
    - {{ shortname }}

"""
