import gradio as gr
from typing import List
import pandas as pd

from app.database.database import SessionLocal
from app.services.dataset_service import DatasetService


def create_dataset_ui():
    """Create Gradio UI for dataset management"""

    with gr.Tab("Dataset Management"):
        gr.Markdown("# 📊 Dataset Management")
        gr.Markdown("Create, view, and manage training datasets")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Create New Dataset")
                dataset_name = gr.Textbox(
                    label="Dataset Name", placeholder="my_dataset"
                )
                dataset_type = gr.Dropdown(
                    choices=["alpaca", "sharegpt", "custom"],
                    label="Dataset Type",
                    value="alpaca",
                )
                dataset_desc = gr.Textbox(
                    label="Description",
                    placeholder="Description of the dataset",
                    lines=3,
                )
                create_dataset_btn = gr.Button("Create Dataset", variant="primary")
                create_output = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### Existing Datasets")
                refresh_datasets_btn = gr.Button("🔄 Refresh")
                datasets_df = gr.Dataframe(
                    headers=["ID", "Name", "Type", "Samples", "Created At"],
                    label="Datasets",
                    interactive=False,
                )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Add Sample to Dataset")
                select_dataset = gr.Dropdown(
                    label="Select Dataset", choices=[], allow_custom_value=False
                )
                sample_instruction = gr.Textbox(
                    label="Instruction",
                    placeholder="What is the capital of France?",
                    lines=2,
                )
                sample_input = gr.Textbox(
                    label="Input (Optional)", placeholder="Additional context", lines=2
                )
                sample_output = gr.Textbox(
                    label="Output",
                    placeholder="The capital of France is Paris.",
                    lines=3,
                )
                sample_system = gr.Textbox(
                    label="System Prompt (Optional)",
                    placeholder="You are a helpful assistant.",
                    lines=2,
                )
                add_sample_btn = gr.Button("Add Sample", variant="primary")
                add_sample_output = gr.Textbox(label="Status", interactive=False)

            with gr.Column():
                gr.Markdown("### View Dataset Samples")
                view_dataset = gr.Dropdown(
                    label="Select Dataset", choices=[], allow_custom_value=False
                )
                view_samples_btn = gr.Button("View Samples")
                samples_df = gr.Dataframe(
                    headers=["ID", "Instruction", "Input", "Output"],
                    label="Samples",
                    interactive=False,
                )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Import/Export Dataset")
                import_dataset = gr.Dropdown(
                    label="Select Dataset for Import", choices=[]
                )
                import_file = gr.File(label="Upload JSON File", file_types=[".json"])
                import_btn = gr.Button("Import from JSON")
                import_output = gr.Textbox(label="Import Status", interactive=False)

            with gr.Column():
                export_dataset = gr.Dropdown(
                    label="Select Dataset for Export", choices=[]
                )
                export_btn = gr.Button("Export to JSON")
                export_file = gr.File(label="Download JSON")
                export_output = gr.Textbox(label="Export Status", interactive=False)

        # Event handlers
        def create_dataset_handler(name, dtype, desc):
            if not name:
                return "Error: Dataset name is required"

            db = SessionLocal()
            try:
                result = DatasetService.create_dataset(db, name, dtype, desc)
                return f"✅ Dataset created: {result['name']} (ID: {result['id']})"
            except Exception as e:
                return f"❌ Error: {str(e)}"
            finally:
                db.close()

        def refresh_datasets_handler():
            db = SessionLocal()
            try:
                datasets = DatasetService.get_datasets(db)
                data = [
                    [
                        ds["id"],
                        ds["name"],
                        ds["type"],
                        ds["total_samples"],
                        ds["created_at"],
                    ]
                    for ds in datasets
                ]
                dataset_names = [ds["name"] for ds in datasets]
                return (
                    data,
                    gr.Dropdown(choices=dataset_names),
                    gr.Dropdown(choices=dataset_names),
                    gr.Dropdown(choices=dataset_names),
                    gr.Dropdown(choices=dataset_names),
                )
            except Exception as e:
                return [], [], [], [], []
            finally:
                db.close()

        def add_sample_handler(dataset_name, instruction, input_text, output, system):
            if not dataset_name or not instruction or not output:
                return "Error: Dataset, instruction, and output are required"

            db = SessionLocal()
            try:
                # Get dataset ID from name
                dataset = DatasetService.get_datasets(db)
                dataset_id = None
                for ds in dataset:
                    if ds["name"] == dataset_name:
                        dataset_id = ds["id"]
                        break

                if not dataset_id:
                    return "Error: Dataset not found"

                DatasetService.add_sample(
                    db, dataset_id, instruction, output, input_text or None, system or None
                )
                return "✅ Sample added successfully"
            except Exception as e:
                return f"❌ Error: {str(e)}"
            finally:
                db.close()

        def view_samples_handler(dataset_name):
            if not dataset_name:
                return []

            db = SessionLocal()
            try:
                datasets = DatasetService.get_datasets(db)
                dataset_id = None
                for ds in datasets:
                    if ds["name"] == dataset_name:
                        dataset_id = ds["id"]
                        break

                if not dataset_id:
                    return []

                samples = DatasetService.get_samples(db, dataset_id)
                data = [
                    [s["id"], s["instruction"], s["input"] or "", s["output"]]
                    for s in samples
                ]
                return data
            except Exception as e:
                return []
            finally:
                db.close()

        def import_handler(dataset_name, file):
            if not dataset_name or not file:
                return "Error: Dataset and file are required"

            db = SessionLocal()
            try:
                datasets = DatasetService.get_datasets(db)
                dataset_id = None
                for ds in datasets:
                    if ds["name"] == dataset_name:
                        dataset_id = ds["id"]
                        break

                if not dataset_id:
                    return "Error: Dataset not found"

                result = DatasetService.import_from_json(db, dataset_id, file.name)
                return f"✅ Imported {result['imported']} of {result['total']} samples"
            except Exception as e:
                return f"❌ Error: {str(e)}"
            finally:
                db.close()

        def export_handler(dataset_name):
            if not dataset_name:
                return None, "Error: Dataset is required"

            db = SessionLocal()
            try:
                datasets = DatasetService.get_datasets(db)
                dataset_id = None
                for ds in datasets:
                    if ds["name"] == dataset_name:
                        dataset_id = ds["id"]
                        break

                if not dataset_id:
                    return None, "Error: Dataset not found"

                output_path = f"/tmp/{dataset_name}.json"
                DatasetService.export_to_json(db, dataset_id, output_path)
                return output_path, f"✅ Dataset exported to {output_path}"
            except Exception as e:
                return None, f"❌ Error: {str(e)}"
            finally:
                db.close()

        # Connect events
        create_dataset_btn.click(
            create_dataset_handler,
            inputs=[dataset_name, dataset_type, dataset_desc],
            outputs=create_output,
        )

        refresh_datasets_btn.click(
            refresh_datasets_handler,
            outputs=[
                datasets_df,
                select_dataset,
                view_dataset,
                import_dataset,
                export_dataset,
            ],
        )

        add_sample_btn.click(
            add_sample_handler,
            inputs=[
                select_dataset,
                sample_instruction,
                sample_input,
                sample_output,
                sample_system,
            ],
            outputs=add_sample_output,
        )

        view_samples_btn.click(
            view_samples_handler, inputs=view_dataset, outputs=samples_df
        )

        import_btn.click(
            import_handler,
            inputs=[import_dataset, import_file],
            outputs=import_output,
        )

        export_btn.click(
            export_handler, inputs=export_dataset, outputs=[export_file, export_output]
        )

    return None
